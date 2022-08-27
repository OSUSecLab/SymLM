# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import encoders


class RobertaHubInterface(nn.Module):
    """A simple PyTorch Hub interface to RoBERTa.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/roberta
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model

        self.bpe = encoders.build_bpe(args)

        # this is useful for determining the device
        self.register_buffer('_float_tensor', torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self, emb_sent_map: dict, *addl_maps, no_separator=False) -> dict:
        """
        BPE-encode a sentence (or multiple sentences).

        Every sequence begins with a beginning-of-sentence (`<s>`) symbol.
        Every sentence ends with an end-of-sentence (`</s>`) and we use an
        extra end-of-sentence (`</s>`) as a separator.

        Example (single sentence): `<s> a b c </s>`
        Example (sentence pair): `<s> d e f </s> </s> 1 2 3 </s>`

        The BPE encoding follows GPT-2. One subtle detail is that the GPT-2 BPE
        requires leading spaces. For example::

            >>> roberta.encode('Hello world').tolist()
            [0, 31414, 232, 2]
            >>> roberta.encode(' world').tolist()
            [0, 232, 2]
            >>> roberta.encode('world').tolist()
            [0, 8331, 2]
        """
        # TODO: this split is not correct
        emb_tok_map = {}
        for emb, sent in emb_sent_map.items():
            emb_tok_map[emb] = '<s> ' + ' '.join([str(_) for _ in sent]) + ' </s>'

        for m in addl_maps:
            for emb, sent in m.items():
                emb_tok_map[emb] += (' </s>' if not no_separator else '')
                emb_tok_map[emb] += ' ' + ' '.join([str(_) for _ in sent]) + ' </s>'

        tokens_map = {}
        for emb, sent in emb_tok_map.items():
            tokens = self.task.source_dictionary[emb].encode_line(sent, append_eos=False, add_if_not_exist=False)
            tokens_map[emb] = tokens.long()

        return tokens_map

    # dict of torch.LongTensor
    def decode(self, emb_tok_map: dict):
        assert list(emb_tok_map.values())[0].dim() == 1
        sentences_map = {}
        for emb, tokens in emb_tok_map.items():
            tokens = tokens.numpy()
            if tokens[0] == self.task.source_dictionary[emb].bos():
                tokens = tokens[1:]  # remove <s>
            eos_mask = (tokens == self.task.source_dictionary[emb].eos())
            doc_mask = eos_mask[1:] & eos_mask[:-1]
            sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
            sentences = [self.task.source_dictionary[emb].string(s) for s in sentences]
            if len(sentences) == 1:
                sentences_map[emb] = sentences[0]
            else:
                sentences_map[emb] = sentences
        return sentences_map

    def extract_features(self, tokens: dict, return_all_hiddens: bool = False) -> torch.Tensor:
        # check if there is only 1 value
        for emb, sent in tokens.items():
            if sent.dim() == 1:
                tokens[emb] = sent.unsqueeze(0)
            if sent.size(-1) > self.model.max_positions():
                raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                    sent.size(-1), self.model.max_positions()
                ))

        features, extra = self.model(
            tokens,
            features_only=False,
            return_all_hiddens=return_all_hiddens,
        )
        return features  # just the last layer's features

    def register_classification_head(
            self, name: str, num_classes: int = None, embedding_size: int = None, **kwargs
    ):
        self.model.register_classification_head(
            name, num_classes=num_classes, embedding_size=embedding_size, **kwargs
        )

    # tokens is dict of  torch.LongTensor
    def predict(self, head: str, tokens: dict, return_logits: bool = False):

        features = self.extract_features({field: tokens[field].to(device=self.device) for field in tokens.keys()})
        print(type(features))
        if (type(features) == dict):
            print(features.keys())
        # logits = self.model.classification_heads[head](features)
        logits = self.model(
            {field: tokens[field].to(device=self.device) for field in tokens.keys()},
            features_only=False,
            return_all_hiddens=False,
        )
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)
        # features = self.extract_features(tokens.to(device=self.device))
        # logits = self.model.classification_heads[head](features) #1 10 768 (1 is for batch size, 10 is expansion, 768 is the token embedding)
        # if return_logits:
        #     return logits
        # return F.log_softmax(logits, dim=-1)

    # TODO: predict masked  - change features only to false

    def extract_features_aligned_to_words(self, sentence: str, return_all_hiddens: bool = False) -> torch.Tensor:
        """Extract RoBERTa features, aligned to spaCy's word-level tokenizer."""
        from fairseq.models.roberta import alignment_utils
        from spacy.tokens import Doc

        nlp = alignment_utils.spacy_nlp()
        tokenizer = alignment_utils.spacy_tokenizer()

        # tokenize both with GPT-2 BPE and spaCy
        bpe_toks = self.encode(sentence)
        spacy_toks = tokenizer(sentence)
        spacy_toks_ws = [t.text_with_ws for t in tokenizer(sentence)]
        alignment = alignment_utils.align_bpe_to_words(self, bpe_toks, spacy_toks_ws)

        # extract features and align them
        features = self.extract_features(bpe_toks, return_all_hiddens=return_all_hiddens)
        features = features.squeeze(0)
        aligned_feats = alignment_utils.align_features_to_words(self, features, alignment)

        # wrap in spaCy Doc
        doc = Doc(
            nlp.vocab,
            words=['<s>'] + [x.text for x in spacy_toks] + ['</s>'],
            spaces=[True] + [x.endswith(' ') for x in spacy_toks_ws[:-1]] + [True, False],
        )
        assert len(doc) == aligned_feats.size(0)
        doc.user_token_hooks['vector'] = lambda token: aligned_feats[token.i]
        return doc

    def fill_mask(self, masked_input: dict, topk: int = 5):
        masked_token = '<mask>'
        # assert masked_token in masked_input and masked_input.count(masked_token) == 1, \
        #     "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(masked_token)
        for emb, sent in masked_input.items():
            if 'byte' not in emb:
                continue
            # assert masked_token in sent and sent.count(masked_token) == 1, \
            assert masked_token in sent, \
                "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(masked_token)

        tokens = self.encode(masked_input)

        masked_index = (tokens['byte1'] == self.task.mask_idx_dict['byte1']).nonzero()
        for emb, sent in tokens.items():
            if sent.dim() == 1:
                tokens[emb] = sent.unsqueeze(0)

        with utils.model_eval(self.model):
            features, extra = self.model(
                {field: tokens[field].to(device=self.device) for field in tokens.keys()},
                features_only=False,
                return_all_hiddens=False,
            )

        topk_filled_outputs_map = {}
        for emb, feature in features.items():
            if 'byte' not in emb:
                continue
            logits = feature[0, masked_index, :].squeeze()
            prob = logits.softmax(dim=0)
            values, index = prob.topk(k=topk, dim=0)
            topk_predicted_token_bpe = self.task.source_dictionary[emb].string(index)
            topk_filled_outputs = []
            for index, predicted_token_bpe in enumerate(topk_predicted_token_bpe.split(' ')):
                predicted_token = predicted_token_bpe
                # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
                # if predicted_token_bpe.startswith('\u2581'):
                #     predicted_token = ' ' + predicted_token
                if " {0}".format(masked_token) in masked_input[emb]:
                    topk_filled_outputs.append((
                        masked_input[emb].replace(
                            ' {0}'.format(masked_token[emb]), predicted_token
                        ),
                        values[index].item(),
                        predicted_token,
                    ))
                else:
                    topk_filled_outputs.append((
                        [predicted_token if tok == masked_token else tok for tok in masked_input[emb]],
                        values[index].item(),
                        predicted_token,
                    ))
            topk_filled_outputs_map[emb] = topk_filled_outputs
        return topk_filled_outputs_map

    # def fill_mask(self, masked_input: str, topk: int = 5):
    #     masked_token = '<mask>'
    #     assert masked_token in masked_input and masked_input.count(masked_token) == 1, \
    #         "Please add one {0} token for the input, eg: 'He is a {0} guy'".format(masked_token)
    #
    #     text_spans = masked_input.split(masked_token)
    #     text_spans_bpe = (' {0} '.format(masked_token)).join(
    #         [self.bpe.encode(text_span.rstrip()) for text_span in text_spans]
    #     ).strip()
    #     print(self.task.source_dictionary)
    #     tokens = self.task.source_dictionary.encode_line(
    #         '<s> ' + text_spans_bpe + ' </s>',
    #         append_eos=False,
    #         add_if_not_exist=False,
    #     )
    #
    #     masked_index = (tokens == self.task.mask_idx).nonzero()
    #     if tokens.dim() == 1:
    #         tokens = tokens.unsqueeze(0)
    #
    #     with utils.model_eval(self.model):
    #         features, extra = self.model(
    #             tokens.long().to(device=self.device),
    #             features_only=False,
    #             return_all_hiddens=False,
    #         )
    #     logits = features[0, masked_index, :].squeeze()
    #     prob = logits.softmax(dim=0)
    #     values, index = prob.topk(k=topk, dim=0)
    #     topk_predicted_token_bpe = self.task.source_dictionary.string(index)
    #
    #     topk_filled_outputs = []
    #     for index, predicted_token_bpe in enumerate(topk_predicted_token_bpe.split(' ')):
    #         predicted_token = self.bpe.decode(predicted_token_bpe)
    #         # Quick hack to fix https://github.com/pytorch/fairseq/issues/1306
    #         if predicted_token_bpe.startswith('\u2581'):
    #             predicted_token = ' ' + predicted_token
    #         if " {0}".format(masked_token) in masked_input:
    #             topk_filled_outputs.append((
    #                 masked_input.replace(
    #                     ' {0}'.format(masked_token), predicted_token
    #                 ),
    #                 values[index].item(),
    #                 predicted_token,
    #             ))
    #         else:
    #             topk_filled_outputs.append((
    #                 masked_input.replace(masked_token, predicted_token),
    #                 values[index].item(),
    #                 predicted_token,
    #             ))
    #     return topk_filled_outputs

    def disambiguate_pronoun(self, sentence: str) -> bool:
        """
        Usage::

            >>> disambiguate_pronoun('The _trophy_ would not fit in the brown suitcase because [it] was too big.')
            True

            >>> disambiguate_pronoun('The trophy would not fit in the brown suitcase because [it] was too big.')
            'The trophy'
        """
        assert hasattr(self.task, 'disambiguate_pronoun'), \
            'roberta.disambiguate_pronoun() requires a model trained with the WSC task.'
        with utils.model_eval(self.model):
            return self.task.disambiguate_pronoun(self.model, sentence, use_cuda=self.device.type == 'cuda')
