from datasets import list_metrics
import evaluate

if __name__ == '__main__':
    # metrics_list = list_metrics()
    metrics_list = evaluate.list_evaluation_modules()
    print(f"{metrics_list}")

    pass
