import train_nets as tn
from lime.lime_tabular import LimeTabularExplainer
from lime.lime_text import LimeTextExplainer



net, X, Y, X_eval, Y_eval = tn.train_speck_distinguisher(50,num_rounds=5,depth=10);


# Create an instance of LimeTabularExplainer
explainer = LimeTabularExplainer(X,
                                 feature_names=[f'feature_{i}' for i in range(X.shape[1])],
                                 class_names=[str(i) for i in range(2)],
                                 )

def predict_fn(data):
    return net.predict(data).astype(float)
# Select an instance to explain
# instance = X_eval[0].reshape(1, -1)

# Explain the instance
explanation = explainer.explain_instance(X_eval[1], predict_fn, top_labels=1)

# Visualize the explanation
# explanation.show_in_notebook(show_table=True, show_all=False)
explanation.save_to_file("Lime.html")
