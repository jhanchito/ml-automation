import gradio as gr
import numpy as np
from model_utils import load_model_bundle

bundle = load_model_bundle()
MODEL = bundle["model"]
TARGET_NAMES = bundle.get("target_names", ["setosa", "versicolor", "virginica"])

# def predict(sepal_length, sepal_width, petal_length, petal_width):
#     X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
#     y = MODEL.predict(X)[0]
#     return TARGET_NAMES[int(y)]

# demo = gr.Interface(
#     fn=predict,
#     inputs=[
#         gr.Number(label="sepal_length"),
#         gr.Number(label="sepal_width"),
#         gr.Number(label="petal_length"),
#         gr.Number(label="petal_width"),
#     ],
#     outputs=gr.Text(label="prediction"),
#     title="ML CI/CT/CD Quickstart (Iris)",
#     allow_flagging="never",
# )


def predict(Account_length,Num_vmail_messages,Tot_day_minutes,Tot_day_calls,Tot_day_charge,Tot_eve_minutes,Tot_eve_calls,Tot_eve_charge,Tot_night_minutes,Tot_night_calls,Tot_night_charge,Tot_intl_minutes,Tot_intl_calls,Tot_intl_charge,Customer_service_calls):
    X = np.array([[Account_length,Num_vmail_messages,Tot_day_minutes,Tot_day_calls,Tot_day_charge,Tot_eve_minutes,Tot_eve_calls,Tot_eve_charge,Tot_night_minutes,Tot_night_calls,Tot_night_charge,Tot_intl_minutes,Tot_intl_calls,Tot_intl_charge,Customer_service_calls]])
    y = MODEL.predict(X)[0]
    return TARGET_NAMES[int(y)]

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Account_length"),
        gr.Number(label="Num_vmail_messages"),
        gr.Number(label="Tot_day_minutes"),
        gr.Number(label="Tot_day_calls"),
        gr.Number(label="Tot_day_charge"),
        gr.Number(label="Tot_eve_minutes"),
        gr.Number(label="Tot_eve_calls"),
        gr.Number(label="Tot_eve_charge"),
        gr.Number(label="Tot_night_minutes"),
        gr.Number(label="Tot_night_calls"),
        gr.Number(label="Tot_night_charge"),
        gr.Number(label="Tot_intl_minutes"),
        gr.Number(label="Tot_intl_calls"),
        gr.Number(label="Tot_intl_charge"),
        gr.Number(label="Customer_service_calls"),
    ],
    outputs=gr.Text(label="prediction"),
    title="ML CI/CT/CD Quickstart (CHURN)",
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
