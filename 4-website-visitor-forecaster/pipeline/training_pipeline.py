from training.process import load_data, prepare_data
from training.train import hyperparameter_tuning, train_and_evaluate, save_model
from utils import *

logger = get_logger(__name__)

def run():
    logger.info("Loading data feature...")
    df = load_data("feature.csv")
    logger.info("Data loaded!")

    logger.info("Preparing data...")
    X, y = prepare_data(df)
    logger.info("Data prepared!")

    logger.info("Hyperparameter Tuning...")
    best_hyperparams = hyperparameter_tuning(X, y)
    logger.info("Hyperparameter Tuning finished!")

    logger.info("Training and evaluating...")
    best_model = train_and_evaluate(X, y, best_hyperparams)
    logger.info("Training and evaluating finished!")

    logger.info("Saving model...")
    save_model(best_model, "best_xgb_model")
    logger.info("Model saved!")

run()
