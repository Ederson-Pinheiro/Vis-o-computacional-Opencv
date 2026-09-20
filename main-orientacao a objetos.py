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
        # Cria a pasta de saída
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    # ==========================================
    # LEITURA DA IMAGEM
    # ==========================================

    def read_img(self, caminho):
        """Lê uma imagem."""
        img = cv.imread(caminho)
        if img is not None:
            return img
        print(f"Erro ao carregar: {caminho}")
        return None

    # ==========================================
    # PRÉ-PROCESSAMENTO
    # ==========================================

    def preprocess(self, img):
        """Converte para escala de cinza e aplica Gaussian Blur."""
        # Grayscale
        img_cinza = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Gaussian Blur
        img_gaussian = cv.GaussianBlur(img_cinza, (5, 5), 0)
        return img_gaussian

    # ==========================================
    # SEGMENTAÇÃO
    # ==========================================

    def segment(self, img):
        """Aplica Threshold Otsu e Canny."""
        # Threshold Otsu
        _, img_otsu = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # Canny
        edges = cv.Canny(img, 50, 150)
        return img_otsu, edges

    # ==========================================
    # MORFOLOGIA
    # ==========================================

    def morphology(self, img):
        """Aplica operação morfológica de abertura."""
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
        img_morph = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
        return img_morph

    # ==========================================
    # REDIMENSIONAMENTO
    # ==========================================

    def resize(self, img):
        """Redimensiona a imagem para 256x256."""
        return cv.resize(img, (256, 256))

    # ==========================================
    # PIPELINE COMPLETO
    # ==========================================

    def process_image(self, img):
        """Executa todas as etapas do processamento."""

        # 1. Pré-processamento
        img_gaussian = self.preprocess(img)
        # 2. Segmentação
        img_otsu, edges = self.segment(img_gaussian)
        # 3. Morfologia
        img_morph = self.morphology(img_otsu)
        # 4. Redimensionamento
        img_resize = self.resize(img_morph)
        return img_resize, edges

    # ==========================================
    # PROCESSAMENTO EM LOTE
    # ==========================================

    def process_batch(self, limit=None):
        """Processa as imagens da pasta de entrada."""

        # Lista os arquivos
        files = [f for f in os.listdir(self.input_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        # Limita a quantidade de imagens
        if limit is not None:
            files = files[:limit]
        # Verifica se existem imagens
        if not files:
            print("Nenhuma imagem encontrada.")
            return

        # Processa cada imagem
        for file in tqdm(files, desc="Processando imagens", unit="img"):
            caminho = os.path.join(self.input_dir, file)
            # Leitura
            img = self.read_img(caminho)
            if img is None:
                continue

            # Processamento
            img_processada, edges = self.process_image(img)
            # Nome sem extensão
            nome = os.path.splitext(file)[0]
            # Caminho de saída
            caminho_saida = os.path.join(self.output_dir, nome + ".png")
            # Salva a imagem processada
            sucesso = cv.imwrite(caminho_saida, img_processada)

            if sucesso:
                print(f"Salva: {caminho_saida}")
            else:
                print(f"Erro ao salvar: {caminho_saida}")

        print("Processamento concluído!")


# ==========================================
# EXECUÇÃO DO PROGRAMA
# ==========================================

processor = ImageProcessor("data/raw_images", "data/processed_images")
processor.process_batch()

