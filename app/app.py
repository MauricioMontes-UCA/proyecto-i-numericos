import gradio as gr
from logistic_regression_model.logistic_regression import classify_flowers
from mult_regression_model.mult_regression import predict_prices

def create_interface():
    
    with gr.Blocks() as app:
        
        gr.Markdown("## 🧠 Aplicación de Modelos Predictivos")

        with gr.Tabs():
            
            with gr.Tab("Regresión Logística"):
                
                gr.Markdown("### 🌼 Clasificación de Flores")
                
                image_input = gr.Image(label="Sube una imagen de flor")
                predict_btn = gr.Button("Clasificar")
                result_text = gr.Textbox(label="Resultado de clasificación", lines=1)
                
                predict_btn.click(classify_flowers, inputs=image_input, outputs=result_text)

            with gr.Tab("Regresión Multivariada"):
                
                gr.Markdown("### 💵 Predicción de Precios")
                
                csv_input = gr.File(file_types=[".csv"], label="Sube archivo CSV")
                predict_csv_btn = gr.Button("Predecir Precios")
                result_table = gr.DataFrame(label="Tabla de precios predichos")
                
                predict_csv_btn.click(predict_prices, inputs=csv_input, outputs=result_table)

    return app

app = create_interface()
app.launch()