from flask import Flask, request, jsonify
from pydantic import BaseModel
from flask_pydantic import validate
import joblib
import pandas as pd

# Criar a instância do Flask
app = Flask(__name__)

# Classe que receberá os dados dos formulários
class request_boyd(BaseModel):
    Genero_Masculino: int
    Idade: int
    Historico_Familiar_Sobrepeso: int
    Consumo_Alta_Caloria_Com_Frequencia: int
    Consumo_Vegetais_Com_Frequencia: int
    Refeicoes_Dia: int
    Consumo_Alimentos_entre_Refeicoes: int
    Fumante: int
    Consumo_Agua: int
    Monitora_Calorias_Ingeridas: int
    Nivel_Atividade_Fisica: int
    Nivel_Uso_Tela: int
    Consumo_Alcool: int
    Transporte_Automovel: int
    Transporte_Bicicleta: int
    Transporte_Motocicleta: int
    Transporte_Publico: int
    Transporte_Caminhada: int

# Carregar o modelo
modelo_obesidade = joblib.load('models/modelo_obesidade.pkl')

@app.route('/prever_obesidade', methods=['POST'])
@validate()
def predict_obesidade(body: request_boyd):
    return