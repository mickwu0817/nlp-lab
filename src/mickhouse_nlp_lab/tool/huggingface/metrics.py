from datasets import list_metrics, load_metric
import evaluate

if __name__ == '__main__':
    # Get Metrics List
    metrics_list1 = list_metrics()
    metrics_list2 = evaluate.list_evaluation_modules()
    print(f"{metrics_list1[:5]}")
    print(f"{metrics_list2[:5]}")


    pass
