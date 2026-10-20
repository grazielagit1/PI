"""
Trabalho 01 de PI - Manipulação de imagens (filtros)

Programa que recebe uma imagem JPG colorida (de qualquer tamanho) e aplica
filtros clássicos da literatura. Todas as convoluções são implementadas
manualmente; a biblioteca Pillow é usada apenas para ler e salvar o arquivo.

Uso:
    python filtros.py imagem.jpg            -> menu interativo
    python filtros.py imagem.jpg sobel      -> aplica um filtro
    python filtros.py imagem.jpg todos      -> aplica todos os filtros

Dependências:
    pip install pillow numpy
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def carregar_imagem(caminho):
    """Lê a imagem e devolve uma matriz RGB (altura x largura x 3) em float."""
    img = Image.open(caminho).convert("RGB")
    return np.asarray(img, dtype=np.float64)


def salvar_imagem(matriz, caminho):
    """Limita os pixels a [0, 255] e salva a imagem."""
    matriz = limitar(matriz).astype(np.uint8)
    Image.fromarray(matriz).save(caminho)


def limitar(matriz):
    """Garante que cada pixel fique no mínimo 0 e no máximo 255."""
    return np.clip(matriz, 0, 255)


def convolucao(imagem, mascara):
    """
    Convolução 2D de uma imagem em tons de cinza com uma máscara 3x3.

    A borda é tratada replicando os pixels da extremidade, para que a imagem
    de saída tenha o mesmo tamanho da entrada.

    Para cada pixel (i, j):
        saida[i, j] = soma de mascara[a, b] * imagem[i + a - 1, j + b - 1]
    """
    altura, largura = imagem.shape
    k = mascara.shape[0]          # tamanho da máscara (3)
    r = k // 2                    # raio da máscara (1)

    # Gira a máscara 180° (definição formal de convolução).
    mascara = np.flipud(np.fliplr(mascara))

    # Replica as bordas.
    expandida = np.pad(imagem, r, mode="edge")

    saida = np.zeros((altura, largura), dtype=np.float64)
    # Em vez de 4 laços aninhados (muito lentos em Python), percorremos só
    # os 9 elementos da máscara e somamos a imagem deslocada multiplicada
    # pelo peso. O resultado é exatamente o mesmo da convolução pixel a pixel.
    for a in range(k):
        for b in range(k):
            saida += mascara[a, b] * expandida[a:a + altura, b:b + largura]
    return saida


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------

def escala_de_cinza(rgb):
    """
    Converte a imagem para cinza usando a média ponderada (luminância)
    recomendada pela ITU-R BT.601: 0.299 R + 0.587 G + 0.114 B.
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


MASCARA_MEDIA = np.ones((3, 3)) / 9.0


def blur(imagem):
    """
    Borramento: convolução com a máscara de média 3x3 (todos 1, vezes 1/9).
    Aceita imagem em cinza (2D) ou colorida (3D); na colorida, a máscara é
    aplicada em cada canal (R, G e B) separadamente.
    """
    if imagem.ndim == 3:
        return np.stack([convolucao(imagem[:, :, c], MASCARA_MEDIA)
                         for c in range(imagem.shape[2])], axis=2)
    return convolucao(imagem, MASCARA_MEDIA)


SOBEL = [
    np.array([[ 1,  2,  1],
              [ 0,  0,  0],
              [-1, -2, -1]], dtype=np.float64),
    np.array([[-1, -2, -1],
              [ 0,  0,  0],
              [ 1,  2,  1]], dtype=np.float64),
    np.array([[ 1,  0, -1],
              [ 2,  0, -2],
              [ 1,  0, -1]], dtype=np.float64),
    np.array([[-1,  0,  1],
              [-2,  0,  2],
              [-1,  0,  1]], dtype=np.float64),
]


def sobel(cinza):
    """
    Detecção de bordas de Sobel: aplica as quatro máscaras, limita cada
    resultado a [0, 255] e soma os resultados (limitando a soma de novo).
    Cada máscara destaca as bordas em um sentido (cima, baixo, esquerda,
    direita); somando as quatro obtemos todas as bordas.
    """
    resultado = np.zeros_like(cinza)
    for mascara in SOBEL:
        resultado += limitar(convolucao(cinza, mascara))
    return limitar(resultado)


def unsharp(imagem):
    """Unsharp: imagem original menos o resultado do blur (detalhes finos)."""
    return imagem - blur(imagem)


def sharpening(imagem):
    """
    Sharpening (nitidez): resultado do unsharp somado à imagem original.
    Recebe a imagem colorida, então a nitidez é aplicada em cada canal
    e as cores são mantidas.
    """
    return imagem + unsharp(imagem)


def negativo(cinza):
    """Filtro extra - negativo: inverte as intensidades (255 - pixel)."""
    return 255 - cinza


# Nome do filtro -> função.
FILTROS = {
    "cinza": lambda rgb: escala_de_cinza(rgb),
    "blur": lambda rgb: blur(rgb),
    "sobel": lambda rgb: sobel(escala_de_cinza(rgb)),
    "unsharp": lambda rgb: unsharp(escala_de_cinza(rgb)),
    "sharpening": lambda rgb: sharpening(rgb),
    "negativo": lambda rgb: negativo(escala_de_cinza(rgb)),
}


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def caminho_saida(caminho_entrada, nome):
    base = os.path.splitext(os.path.basename(caminho_entrada))[0]
    pasta = os.path.dirname(os.path.abspath(caminho_entrada))
    return os.path.join(pasta, f"{base}_{nome}.png")


def aplicar(rgb, nome, caminho_entrada):
    """Aplica um filtro, salva o arquivo e devolve a matriz resultante."""
    resultado = limitar(FILTROS[nome](rgb))
    saida = caminho_saida(caminho_entrada, nome)
    salvar_imagem(resultado, saida)
    print(f"[ok] {nome:<11} -> {saida}")
    return resultado


def aplicar_todos(rgb, caminho_entrada):
    """
    Aplica todos os filtros (um arquivo para cada) e também gera
    <imagem>_todos.png: um mosaico com a original e todos os resultados
    lado a lado, com o nome de cada filtro, para facilitar a comparação.
    """
    paineis = [("original", rgb)]
    for nome in FILTROS:
        paineis.append((nome, aplicar(rgb, nome, caminho_entrada)))

    largura_painel = 500
    altura_img, largura_img = rgb.shape[:2]
    altura_painel = round(altura_img * largura_painel / largura_img)
    faixa = 30                                   # espaço para o título
    colunas = 4
    linhas = (len(paineis) + colunas - 1) // colunas

    mosaico = Image.new("RGB", (colunas * largura_painel,
                                linhas * (altura_painel + faixa)), "white")
    desenho = ImageDraw.Draw(mosaico)
    try:
        fonte = ImageFont.load_default(size=22)
    except TypeError:                            # Pillow antigo
        fonte = ImageFont.load_default()

    for i, (nome, matriz) in enumerate(paineis):
        img = Image.fromarray(limitar(matriz).astype(np.uint8)).convert("RGB")
        img = img.resize((largura_painel, altura_painel))
        x = (i % colunas) * largura_painel
        y = (i // colunas) * (altura_painel + faixa)
        desenho.text((x + 8, y + 4), nome, fill="black", font=fonte)
        mosaico.paste(img, (x, y + faixa))

    saida = caminho_saida(caminho_entrada, "todos")
    mosaico.save(saida)
    print(f"[ok] {'todos':<11} -> {saida}")


def menu(rgb, caminho):
    nomes = list(FILTROS)
    while True:
        print("\nEscolha uma transformação:")
        for i, nome in enumerate(nomes, 1):
            print(f"  {i}) {nome}")
        print(f"  {len(nomes) + 1}) todos")
        print("  0) sair")
        opcao = input("> ").strip().lower()
        if opcao in ("0", "sair"):
            break
        if opcao in ("todos", str(len(nomes) + 1)):
            aplicar_todos(rgb, caminho)
        elif opcao in nomes:
            aplicar(rgb, opcao, caminho)
        elif opcao.isdigit() and 1 <= int(opcao) <= len(nomes):
            aplicar(rgb, nomes[int(opcao) - 1], caminho)
        else:
            print("Opção inválida.")


def main():
    if len(sys.argv) >= 2:
        caminho = sys.argv[1]
    else:
        # Rodando sem argumentos (ex.: botão "Run" da IDE): pergunta a imagem.
        caminho = input("Caminho da imagem JPG: ").strip().strip('"')

    rgb = carregar_imagem(caminho)
    print(f"Imagem carregada: {caminho} ({rgb.shape[1]}x{rgb.shape[0]})")

    if len(sys.argv) <= 2:
        menu(rgb, caminho)
        return

    escolha = sys.argv[2].lower()
    if escolha == "todos":
        aplicar_todos(rgb, caminho)
    elif escolha in FILTROS:
        aplicar(rgb, escolha, caminho)
    else:
        print(f"Filtro desconhecido: {escolha}. Opções: {', '.join(FILTROS)}, todos")
        sys.exit(1)


if __name__ == "__main__":
    main()