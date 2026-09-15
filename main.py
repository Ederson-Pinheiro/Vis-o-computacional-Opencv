import cv2 as cv
import os
from pathlib import Path
from tqdm import tqdm


class ImageProcessor:
    """Classe responsável pelo processamento das imagens."""

    def __init__(self, input_dir, output_dir):
        """Define as pastas de entrada e saída."""

        self.input_dir = input_dir
        self.output_dir = output_dir

    def preprocess(self, img):
        """Aplica grayscale e Gaussian Blur."""

        # Grayscale
        img_cinza = cv.cvtColor(
            img,
            cv.COLOR_BGR2GRAY
        )

        # Gaussian Blur
        img_gaussian = cv.GaussianBlur(
            img_cinza,
            (5, 5),
            0
        )

        return img_gaussian

    def segment(self, img):
        """Aplica Threshold Otsu e Canny."""

        # Threshold Otsu
        _, img_otsu = cv.threshold(
            img,
            0,
            255,
            cv.THRESH_BINARY + cv.THRESH_OTSU
        )

        # Canny
        edges = cv.Canny(
            img,
            50,
            150
        )

        return img_otsu, edges

    def morphology(self, img):
        """Aplica operação morfológica."""

        kernel = cv.getStructuringElement(
            cv.MORPH_RECT,
            (3, 3)
        )

        img_morph = cv.morphologyEx(
            img,
            cv.MORPH_OPEN,
            kernel
        )

        return img_morph

    def resize(self, img):
        """Redimensiona a imagem para 256x256."""

        return cv.resize(
            img,
            (256, 256)
        )

    def process_image(self, img):
        """Executa todo o pipeline."""

        # Grayscale + Gaussian Blur
        img_gaussian = self.preprocess(img)

        # Otsu + Canny
        img_otsu, edges = self.segment(img_gaussian)

        # Morfologia
        img_morph = self.morphology(img_otsu)

        # Resize
        img_resize = self.resize(img_morph)

        return img_resize, edges

    def process_batch(self):
        """Processa todas as imagens da pasta."""

        # Cria a pasta de saída
        Path(self.output_dir).mkdir(
            parents=True,
            exist_ok=True
        )

        # Lista as imagens
        files = [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp")
            )
        ]

        if not files:
            print("Nenhuma imagem encontrada.")
            return

        # Processa as imagens
        for file in tqdm(
            files,
            desc="Processando imagens",
            unit="img"
        ):

            caminho = os.path.join(
                self.input_dir,
                file
            )

            img = cv.imread(caminho)

            if img is None:
                print(f"Erro ao carregar: {file}")
                continue

            # Processamento
            img_processada, edges = self.process_image(img)

            # Nome do arquivo
            nome = os.path.splitext(file)[0]

            # Caminho de saída
            caminho_saida = os.path.join(
                self.output_dir,
                nome + ".png"
            )

            # Salva a imagem
            sucesso = cv.imwrite(
                caminho_saida,
                img_processada
            )

            if sucesso:
                print(f"Salva: {caminho_saida}")
            else:
                print(f"Erro ao salvar: {caminho_saida}")

        print("Processamento concluído!")


# ==========================================
# EXECUÇÃO DO PROGRAMA
# ==========================================

processor = ImageProcessor(
    "data/raw_images",
    "data/processed_images"
)

processor.process_batch()

