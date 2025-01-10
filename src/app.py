import streamlit as st
import requests
import pandas as pd
import plotly.express as px
from io import BytesIO

def main():
    st.title("Detector de Logos en Videos")
    
    # Subida de video
    video_file = st.file_uploader("Subir video", type=['mp4', 'avi', 'mov'])
    
    if video_file is not None:
        # Mostrar progreso
        with st.spinner('Procesando video...'):
            files = {'video': video_file}
            response = requests.post(
                "http://localhost:8000/detect-logos/",
                files=files
            )
            
            if response.status_code == 200:
                results = response.json()
                
                # Mostrar estadísticas generales
                st.header("Estadísticas de Detección")
                stats_df = pd.DataFrame.from_dict(
                    results['statistics'],
                    orient='index'
                )
                st.dataframe(stats_df)
                
                # Gráfico de tiempo en pantalla
                fig = px.bar(
                    stats_df,
                    y='time_percentage',
                    title='Porcentaje de Tiempo en Pantalla por Logo'
                )
                st.plotly_chart(fig)
                
                # Timeline de detecciones
                st.header("Timeline de Detecciones")
                timeline_data = []
                for det in results['detections']:
                    timeline_data.append({
                        'tiempo': det['time'],
                        'logo': det['class'],
                        'confianza': det['confidence']
                    })
                
                timeline_df = pd.DataFrame(timeline_data)
                fig = px.scatter(
                    timeline_df,
                    x='tiempo',
                    y='confianza',
                    color='logo',
                    title='Detecciones a lo largo del video'
                )
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()