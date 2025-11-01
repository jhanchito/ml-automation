from train import train_and_eval
#from train import train_and_eval_custom 

def test_training_runs():
    #metrics = train_and_eval()
    metrics = train_and_eval()
    assert "accuracy" in metrics and metrics["accuracy"] > 0.5
