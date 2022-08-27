from inspect import ArgSpec
from gensim.models import Word2Vec, FastText
from gensim import utils
from gensim.utils import tokenize
from gensim.test.utils import datapath, get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data-path", help="file path to the training data", type=str)
    parser.add_argument("--model-name", help="name of the model to be trained, options: ['skip_gram', 'cbow', 'fasttext']", type=str, default='cbow')
    parser.add_argument("--num-epochs", help="number of epochs to train the model", type=int, default=50)
    args = parser.parse_args()
    training_set = args.training_data_path
    train_one_model(args.model_name, training_set, args.num_epochs)

def load_model(model_name, num_epochs):
    model_path = f"models/{model_name}_{num_epochs}.model"
    if model_name == "fasttext":
        model = FastText.load(model_path)
    else:
        model = Word2Vec.load(model_path)
    return model

class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self):
        print(f"epoch: {self.epoch}")
        self.epoch += 1


def train_one_model(model_name, training_set, num_epochs):

    def create_model(model_name):
        if model_name == "skip_gram":
            model = Word2Vec(size=100, window=5, min_count=1, workers=30, negative=5, sg=1)
        elif model_name == "cbow":
            model = Word2Vec(size=100, window=5, min_count=1, workers=30, negative=5, sg=0)
        elif model_name == "fasttext":
            model = FastText(size=100, window=5, min_count=1, workers=30, negative=5, sg=1)
        else:
            model = None
        return model

    corpus_file = datapath(training_set)
    
    print(f"Train {model_name} with {num_epochs} epochs")
    print(f"Create {model_name}")
    model = create_model(model_name)
    model.build_vocab(corpus_file=corpus_file)
    total_words = model.corpus_total_words
    print("# total words:", total_words)

    model.train(corpus_file=corpus_file, total_words=total_words, epochs=num_epochs, report_delay=1,
                compute_loss=True,  # set compute_loss = True
                callbacks=[callback()])
    model_path = f"models/{model_name}_{num_epochs}.model"

    model.save(model_path)
    print(f"Save {model_name} to {model_path}")


if __name__ == "__main__":
    main()
