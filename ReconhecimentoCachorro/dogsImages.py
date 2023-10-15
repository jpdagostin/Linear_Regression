from google_images_download import google_images_download

# Palavra-chave para pesquisa de imagens
keyword = "cachorro"

# Número de imagens para baixar
limit = 100

# Diretório para salvar as imagens
diretorio_destino = "C:\\Users\\dagos\\OneDrive\\Desktop\\Aprendizado_de-maquina\\Utils\\BuscaBinaria\\ReconhecimentoCachorro\\cachorros"

# Cria um objeto da classe google_images_download
response = google_images_download.googleimagesdownload()

# Configura os parâmetros de pesquisa
arguments = {
    "keywords": keyword,
    "limit": limit,
    "output_directory": diretorio_destino
}

# Realiza o download das imagens
response.download(arguments)
