from remove_bg import extract_background_foreground
from utils import load_images, show_results

def main():
    # Caminho da pasta de imagens
    image_folder = "images"

    # Carregar a sequência de imagens
    images = load_images(image_folder)

    # Extrair background e foreground
    background, foregrounds = extract_background_foreground(images, rank=1)

    # Exibir tudo
    show_results(images, background, foregrounds)

if __name__ == "__main__":
    main()
