pipeline {
    agent any

    triggers {
        githubPush()
    }

    environment {
        REPO_URL     = 'https://github.com/botanicalex/PYTHON_ETL'
        PYTHON       = 'python3'
        NOTIFY_EMAIL = 'alexavascolopera@gmail.com'
    }

    stages {

        stage('Clonar repositorio') {
            steps {
                echo '📥 Clonando repositorio PYTHON_ETL...'
                checkout([
                    $class: 'GitSCM',
                    branches: [[name: '*/master']],
                    userRemoteConfigs: [[url: "${REPO_URL}"]]
                ])
                echo "✅ Repositorio clonado: ${env.GIT_BRANCH}"
            }
        }

        stage('Verificar estructura del proyecto') {
            steps {
                echo '🔍 Verificando archivos requeridos...'
                script {
                    def archivos = [
                        'mlops_pipeline/src/ft_engineering.py',
                        'mlops_pipeline/src/model_training.py',
                        'mlops_pipeline/src/model_deploy.py',
                        'mlops_pipeline/src/model_evaluation.py',
                        'mlops_pipeline/src/model_monitoring.py',
                        'config.json',
                        'requirements.txt',
                        'readme.md'
                    ]
                    def faltantes = []
                    archivos.each { archivo ->
                        if (!fileExists(archivo)) {
                            faltantes.add(archivo)
                            echo "❌ FALTA: ${archivo}"
                        } else {
                            echo "✅ OK: ${archivo}"
                        }
                    }
                    if (faltantes.size() > 0) {
                        error("❌ Faltan archivos: ${faltantes.join(', ')}")
                    }
                    echo '✅ Estructura verificada correctamente.'
                }
            }
        }

        stage('Validar scripts Python') {
            steps {
                echo '🐍 Validando sintaxis Python...'
                sh """
                    python3 -m py_compile mlops_pipeline/src/ft_engineering.py   && echo '✅ ft_engineering.py OK'
                    python3 -m py_compile mlops_pipeline/src/model_training.py   && echo '✅ model_training.py OK'
                    python3 -m py_compile mlops_pipeline/src/model_deploy.py     && echo '✅ model_deploy.py OK'
                    python3 -m py_compile mlops_pipeline/src/model_monitoring.py && echo '✅ model_monitoring.py OK'
                    python3 -m py_compile mlops_pipeline/src/model_evaluation.py && echo '✅ model_evaluation.py OK'
                """
            }
        }

        stage('Validar config.json') {
            steps {
                echo '⚙️ Validando config.json...'
                script {
                    def resultado = sh(
                        script: """
                            python3 -c "
import json, sys
with open('config.json') as f:
    cfg = json.load(f)
claves = ['project_name','github_repo','data_file','model_file','target',
          'test_size','random_state','api_host','api_port',
          'monitoring_period_days','drift_threshold']
faltantes = [c for c in claves if c not in cfg]
if faltantes:
    print('FALTANTES:', faltantes)
    sys.exit(1)
print('OK')
"
                        """,
                        returnStdout: true
                    ).trim()
                    if (!resultado.contains('OK')) {
                        error("❌ config.json incompleto.")
                    }
                    echo '✅ config.json validado.'
                }
            }
        }
    }

    post {
        success {
            mail(
                to: "${NOTIFY_EMAIL}",
                subject: "✅ Jenkins OK — PYTHON_ETL Build #${env.BUILD_NUMBER}",
                body: "El pipeline finalizó exitosamente.\n\nBuild: #${env.BUILD_NUMBER}\nURL: ${env.BUILD_URL}"
            )
        }
        failure {
            mail(
                to: "${NOTIFY_EMAIL}",
                subject: "❌ Jenkins FALLÓ — PYTHON_ETL Build #${env.BUILD_NUMBER}",
                body: "El pipeline encontró un error.\n\nBuild: #${env.BUILD_NUMBER}\nURL: ${env.BUILD_URL}"
            )
        }
        always {
            echo '🏁 Pipeline finalizado.'
        }
    }
}
