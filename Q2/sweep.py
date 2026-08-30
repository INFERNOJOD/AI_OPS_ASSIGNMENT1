
import warnings
warnings.filterwarnings("ignore")
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, log_loss, f1_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mnist-mlp")

print("Loading MNIST dataset...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X, y = mnist.data, mnist.target.astype(int)

rng = np.random.RandomState(42)
idx = rng.choice(len(X), size=10000, replace=False)
X, y = X[idx], y[idx]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
CLASSES = np.unique(y_train)

def train_and_log(lr, bs):
    with mlflow.start_run(run_name=f"mlp-lr{lr}-bs{bs}"):
        mlflow.log_param("learning_rate_init", lr)
        mlflow.log_param("batch_size", bs)
        mlflow.log_param("model_type", "MLPClassifier")
        
        model = MLPClassifier(hidden_layer_sizes=(100,), learning_rate_init=lr, 
                              batch_size=bs, solver="adam", random_state=42, 
                              max_iter=1, warm_start=True)
        
        for epoch in range(15):
            model.partial_fit(X_train, y_train, classes=CLASSES)
            train_loss = log_loss(y_train, model.predict_proba(X_train), labels=CLASSES)
            val_accuracy = accuracy_score(y_test, model.predict(X_test))
            
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            
        final_preds = model.predict(X_test)
        mlflow.log_metric("accuracy", accuracy_score(y_test, final_preds))
        mlflow.log_metric("f1_macro", f1_score(y_test, final_preds, average="macro"))
        
        mlflow.sklearn.log_model(
            model, 
            artifact_path="model", 
            skops_trusted_types=["sklearn.neural_network._stochastic_optimizers.AdamOptimizer"]
        )
        print(f"Finished training for lr={lr} and batch={bs}")

for lr in [0.0001, 0.001, 0.01]:
    for bs in [32, 128]:
        train_and_log(lr, bs)