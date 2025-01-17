# run.py
import subprocess
import sys
import os
from dotenv import load_dotenv
import time
from app.database import init_db
import webbrowser
import signal
import psutil

load_dotenv()

def kill_process_on_port(port):
    for proc in psutil.process_iter(['pid', 'name', 'net_connections']):
        try:
            for conn in proc.net_connections():
                if conn.laddr.port == port:
                    os.kill(proc.pid, signal.SIGTERM)
                    time.sleep(1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def main():
    # Inicializar la base de datos
    print("Inicializando base de datos...")
    init_db()
    
    # Matar procesos que puedan estar usando los puertos necesarios
    kill_process_on_port(8000)  # Puerto API
    kill_process_on_port(8501)  # Puerto Streamlit
    
    # Definir comandos
    api_command = [sys.executable, "-m", "uvicorn", "app.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
    streamlit_command = [sys.executable,"-m", "streamlit", "run", "frontend/streamlit_app.py"]
    
    try:
        # Iniciar API
        print("Iniciando API...")
        api_process = subprocess.Popen(api_command)
        
        # Esperar un momento para asegurarse de que la API está en funcionamiento
        time.sleep(2)
        
        # Iniciar Streamlit
        print("Iniciando Streamlit...")
        streamlit_process = subprocess.Popen(streamlit_command)
        
        # Abrir navegador automáticamente
        time.sleep(3)
        webbrowser.open('http://localhost:8501')  # Streamlit UI
        webbrowser.open('http://localhost:8000/docs')  # API docs
        
        print("\nAplicación iniciada correctamente!")
        print("Frontend: http://localhost:8501")
        print("API docs: http://localhost:8000/docs")
        print("\nPresiona Ctrl+C para detener la aplicación...")
        
        # Mantener el script ejecutándose
        api_process.wait()
        streamlit_process.wait()
        
    except KeyboardInterrupt:
        print("\nDeteniendo la aplicación...")
        
        # Detener procesos
        api_process.terminate()
        streamlit_process.terminate()
        
        # Esperar a que los procesos terminen
        api_process.wait()
        streamlit_process.wait()
        
        print("Aplicación detenida correctamente!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
        # Asegurarse de que los procesos se detengan en caso de error
        try:
            api_process.terminate()
            streamlit_process.terminate()
        except:
            pass
        
        sys.exit(1)

if __name__ == "__main__":
    main()