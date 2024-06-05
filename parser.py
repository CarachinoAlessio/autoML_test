import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--target_column", type=str, default=None, help='max train time in seconds', required=True)
    parser.add_argument("--metric", type=str, default='roc_auc', choices=['roc_auc'], help='')
    parser.add_argument("--autoML", type=str, default='Autogluon', choices=['Autogluon', 'H2O', 'Autosklearn'], help='')
    parser.add_argument("--xai", type=str, default='shap', choices=['shap', 'lime'], help='')
    parser.add_argument("--max_train_time", type=int, default=None, help='max train time in seconds')
    parser.add_argument("--task", type=str, default='classification', choices=['classification', 'regression'], help='', required=True)
    parser.add_argument("--problem_type", type=str, default='binary', choices=['binary', 'multiclass', 'quantile', 'regression'], help='')

    parser.add_argument("--num_gpus", type=int, default=0, help='da usare solo con autoogluon')
    parser.add_argument("--autogluon_preset", type=str, default='medium_quality', choices=['medium_quality', 'best_quality'], help='')
    parser.add_argument("--max_mem_size", type=int, default=None, help="Numero di GB")

    args = parser.parse_args()
    return args
