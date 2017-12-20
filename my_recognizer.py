import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for _, (X, lengths) in test_set.get_all_Xlengths().items():
        prob_dict = {}
        best_logL = float("-Inf")
        best_guess = ""
        for trained_word, model in models.items():
            try: # There is a bug in hlmmlearn, so it must catch exception.
                logL = model.score(X, lengths)
                prob_dict[trained_word] = logL

            except Exception:
                prob_dict[trained_word] = float("-Inf")

            if logL > best_logL:
                best_logL = logL
                best_guess = trained_word

        probabilities.append(prob_dict)
        guesses.append(best_guess)

    return probabilities, guesses
