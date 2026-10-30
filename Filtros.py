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

# FUNÇÕES AUXILIARES
def carregar_imagem(caminho):
    """Lê a imagem e devolve uma matriz RGB (altura x largura x 3) em float."""
    # .convert("RGB") garante 3 canais mesmo se o arquivo vier em outro
    # formato (ex.: PNG com transparência ou imagem já em tons de cinza).
    img = Image.open(caminho).convert("RGB")
    # Convertemos para float64 (número com casas decimais) porque os filtros
    # geram valores negativos, maiores que 255 e fracionários (ex.: 1/9).
    # Se ficássemos em uint8 (0 a 255 inteiro), as contas "dariam a volta"
    # (ex.: 250 + 10 viraria 4) e o resultado sairia errado.
    return np.asarray(img, dtype=np.float64)


def salvar_imagem(matriz, caminho):
    """Limita os pixels a [0, 255] e salva a imagem."""
    # Um arquivo de imagem só aceita inteiros de 0 a 255 (uint8), então
    # primeiro cortamos os valores fora da faixa e depois convertemos o tipo.
    matriz = limitar(matriz).astype(np.uint8)
    # fromarray transforma a matriz NumPy de volta em imagem do Pillow.
    # O formato (PNG) é escolhido pela extensão do nome do arquivo.
    Image.fromarray(matriz).save(caminho)


def limitar(matriz):
    """Garante que cada pixel fique no mínimo 0 e no máximo 255."""
    # np.clip troca tudo que for < 0 por 0 e tudo que for > 255 por 255.
    # É o "limitar a no mínimo 0 e no máximo 255" pedido no enunciado.
    return np.clip(matriz, 0, 255)


def convolucao(imagem, mascara):
    """
    A convolução passa uma máscara 3x3 por cima de cada pixel.
    O novo valor do pixel é a soma dos 9 vizinhos, cada um multiplicado pelo peso que está na mesma posição da máscara
    """
    # shape de uma imagem em cinza = (número de linhas, número de colunas).
    altura, largura = imagem.shape
    k = mascara.shape[0]          # tamanho da máscara (3)
    r = k // 2                    # raio da máscara (1): quantos vizinhos para cada lado

    # Gira a máscara 180° (definição formal de convolução).
    # fliplr espelha esquerda/direita e flipud espelha cima/baixo.
    # Sem o giro, a operação se chamaria "correlação". Para a máscara de
    # média não faz diferença (ela é simétrica), mas para o Sobel faz: o giro
    # troca o sinal do resultado. Como aplicamos as duas versões de cada
    # máscara do Sobel (com sinais opostos), o resultado final é o mesmo.
    mascara = np.flipud(np.fliplr(mascara))

    # Replica as bordas.
    # Os pixels da borda não têm todos os 8 vizinhos. np.pad acrescenta uma
    # moldura de r pixels em volta da imagem, copiando o pixel mais próximo
    # (mode="edge"). Assim todo pixel tem vizinhança 3x3 completa e a saída
    # fica do mesmo tamanho da entrada.
    expandida = np.pad(imagem, r, mode="edge")

    # Matriz de zeros onde vamos acumular a soma ponderada de cada pixel.
    saida = np.zeros((altura, largura), dtype=np.float64)

    # Em vez de 4 laços aninhados (muito lentos em Python), percorremos só
    # os 9 elementos da máscara e somamos a imagem deslocada multiplicada
    # pelo peso. O resultado é exatamente o mesmo da convolução pixel a pixel.
    #
    # Exemplo: quando a=0 e b=0 (canto superior esquerdo da máscara),
    # expandida[0:altura, 0:largura] é a imagem "deslocada" de forma que,
    # na posição (i, j), está o vizinho de cima à esquerda do pixel (i, j).
    # Multiplicamos esse vizinho pelo peso mascara[0, 0] e somamos na saída.
    # Repetindo para as 9 posições, cada pixel da saída recebe a soma dos
    # 9 vizinhos vezes seus pesos, que é a definição da convolução.
    for a in range(k):            # linha da máscara (0, 1, 2)
        for b in range(k):        # coluna da máscara (0, 1, 2)
            saida += mascara[a, b] * expandida[a:a + altura, b:b + largura]
    return saida


# ===========================================================================
# FILTROS
# ===========================================================================

def escala_de_cinza(rgb):
    """
    Converte a imagem para cinza usando a média ponderada (luminância)
    recomendada pela ITU-R BT.601: 0.299 R + 0.587 G + 0.114 B.
    """
    # Separa os 3 canais. rgb[:, :, 0] = todas as linhas, todas as colunas,
    # canal 0 (vermelho). Cada um vira uma matriz altura x largura.
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    # Os pesos são diferentes porque o olho humano é mais sensível ao verde
    # e menos ao azul. Uma média simples (R+G+B)/3 também funcionaria, mas
    # deixaria, por exemplo, um azul forte claro demais.
    # O resultado é UMA matriz (um único canal): a imagem em cinza.
    return 0.299 * r + 0.587 * g + 0.114 * b


# Máscara de média 3x3 do enunciado: todos os elementos valem 1 e a máscara
# é multiplicada por 1/9. Assim cada pixel vira a média dele com os 8
# vizinhos. np.ones((3, 3)) cria uma matriz 3x3 cheia de 1.
MASCARA_MEDIA = np.ones((3, 3)) / 9.0


def passadas_blur(cinza):
    """
    Uma única passada da máscara 3x3 só mistura cada pixel com os vizinhos
    imediatos; numa foto grande (ex.: 4000x3000) isso é invisível a olho nu.
    Por isso aplicamos a mesma máscara várias vezes, proporcional ao tamanho
    da imagem (cerca de 1 passada a cada 50 pixels do maior lado).
    """
    # max(cinza.shape) = o maior lado da imagem (altura ou largura).
    # // é divisão inteira. Exemplo: 4000 // 50 = 80 passadas.
    # O max(3, ...) garante pelo menos 3 passadas em imagens pequenas.
    return max(3, max(cinza.shape) // 50)


def blur(cinza, passadas=None):
    """
    Borramento: convolução com a máscara de média 3x3 (todos 1, vezes 1/9),
    aplicada 'passadas' vezes seguidas para o efeito ficar visível.
    """
    # Se ninguém informou o número de passadas, calcula pelo tamanho.
    if passadas is None:
        passadas = passadas_blur(cinza)

    # Unsharp e sharpening também usam o blur. Guardamos o último resultado
    # para não refazer todas as passadas quando a mesma imagem chega de novo.
    # (No modo "todos", isso faz o programa rodar cerca de 2x mais rápido.)
    ultimo = _cache_blur.get("entrada")   # imagem do último blur (ou None)
    if (ultimo is not None                       # já existe um blur guardado
            and _cache_blur["passadas"] == passadas   # com as mesmas passadas
            and ultimo.shape == cinza.shape           # do mesmo tamanho
            and np.array_equal(ultimo, cinza)):       # e com os mesmos pixels
        return _cache_blur["saida"]               # reaproveita o resultado

    # Aplica a máscara de média repetidas vezes: a saída de uma passada é a
    # entrada da próxima. Cada passada espalha a mistura por mais 1 pixel.
    resultado = cinza
    for _ in range(passadas):     # "_" = variável do laço que não usamos
        resultado = convolucao(resultado, MASCARA_MEDIA)

    # Guarda entrada e saída para a próxima chamada. .copy() evita que uma
    # alteração futura na matriz original mude o que ficou guardado.
    _cache_blur.update(entrada=cinza.copy(), passadas=passadas, saida=resultado)
    return resultado


# Dicionário que guarda o último blur calculado (usado pela função acima).
# Começa vazio; é preenchido na primeira vez que o blur roda.
_cache_blur = {}


# As quatro máscaras de Sobel, exatamente como no enunciado.
#   1ª e 2ª: gradiente VERTICAL (comparam a linha de cima com a de baixo),
#            destacam bordas horizontais. Uma é o negativo da outra.
#   3ª e 4ª: gradiente HORIZONTAL (comparam a coluna da esquerda com a da
#            direita), destacam bordas verticais. Uma é o negativo da outra.
# O "2" no meio dá mais peso ao vizinho mais próximo do centro.
# Numa região lisa (todos os pixels iguais) a soma dos pesos é 0, então o
# resultado é 0 (preto). Onde há uma mudança brusca de cor, o valor é alto.
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
    # Começa com uma matriz de zeros do mesmo tamanho da imagem.
    resultado = np.zeros_like(cinza)
    for mascara in SOBEL:
        # Cada máscara dá valores positivos para bordas num sentido
        # (ex.: claro em cima e escuro embaixo) e negativos no sentido oposto.
        # Limitar corta os negativos para 0, ficando só o "seu" sentido.
        # Como a máscara seguinte é a oposta, ela pega o sentido que foi
        # cortado. Somando as quatro, nenhuma borda se perde.
        resultado += limitar(convolucao(cinza, mascara))
    # A soma pode passar de 255, então limitamos de novo.
    return limitar(resultado)


def unsharp(cinza):
    """Unsharp: imagem original menos o resultado do blur (detalhes finos)."""
    # O blur remove os detalhes finos (bordas, texturas). Tirando o blur da
    # original, sobra justamente o que o blur removeu: os detalhes.
    # Em regiões lisas, original e blur são quase iguais, então o resultado
    # fica perto de 0. Perto das bordas fica positivo de um lado e negativo
    # do outro. (A função aplicar soma 128 antes de salvar; veja lá.)
    return cinza - blur(cinza)


def sharpening(cinza):
    """Sharpening (nitidez): resultado do unsharp somado à imagem original."""
    # Original + detalhes = original com os detalhes reforçados.
    # As bordas ficam mais marcadas e a imagem parece mais nítida.
    # Equivale a: 2 * original - blur.
    return cinza + unsharp(cinza)


def negativo(cinza):
    """Filtro extra - negativo: inverte as intensidades (255 - pixel)."""
    # Preto (0) vira branco (255), branco vira preto e o resto se inverte.
    # Com NumPy, 255 - cinza já faz a conta para todos os pixels de uma vez.
    return 255 - cinza


# Tabela "nome do filtro -> função que aplica o filtro".
# Cada lambda é uma função curta que recebe a imagem colorida (rgb),
# converte para cinza (o enunciado permite) e aplica o filtro.
# Com essa tabela, o menu e a opção "todos" não precisam de um if para cada
# filtro: basta percorrer o dicionário. Para adicionar um filtro novo,
# é só criar a função e acrescentar uma linha aqui.
# O unsharp é ajustado na hora de salvar (ver aplicar) para ficar visível,
# já que seus valores ficam perto de 0.
FILTROS = {
    "cinza": lambda rgb: escala_de_cinza(rgb),
    "blur": lambda rgb: blur(escala_de_cinza(rgb)),
    "sobel": lambda rgb: sobel(escala_de_cinza(rgb)),
    "unsharp": lambda rgb: unsharp(escala_de_cinza(rgb)),
    "sharpening": lambda rgb: sharpening(escala_de_cinza(rgb)),
    "negativo": lambda rgb: negativo(escala_de_cinza(rgb)),
}


# ===========================================================================
# PROGRAMA PRINCIPAL
# ===========================================================================

def caminho_saida(caminho_entrada, nome):
    """Monta o nome do arquivo de saída: foto.jpg + 'blur' -> foto_blur.png."""
    # basename tira as pastas ("C:/fotos/foto.jpg" -> "foto.jpg") e
    # splitext separa a extensão ("foto.jpg" -> ("foto", ".jpg")).
    base = os.path.splitext(os.path.basename(caminho_entrada))[0]
    # Pasta onde está a imagem original: a saída é salva ao lado dela.
    pasta = os.path.dirname(os.path.abspath(caminho_entrada))
    # Salvamos em PNG porque ele não perde qualidade (o JPG comprime e
    # criaria pequenos defeitos em cima do resultado do filtro).
    return os.path.join(pasta, f"{base}_{nome}.png")


def aplicar(rgb, nome, caminho_entrada):
    """Aplica um filtro, salva o arquivo e devolve a matriz resultante."""
    # Busca a função do filtro na tabela FILTROS e a executa.
    resultado = FILTROS[nome](rgb)
    if nome == "unsharp":
        # Os valores do unsharp ficam em torno de 0 (podem ser negativos).
        # Se salvássemos assim, os negativos virariam 0 e a imagem sairia
        # quase toda preta. Somamos 128 para que o cinza médio represente
        # "sem detalhe", o claro um detalhe positivo e o escuro um negativo.
        resultado = resultado + 128
    resultado = limitar(resultado)            # garante a faixa 0 a 255
    saida = caminho_saida(caminho_entrada, nome)
    salvar_imagem(resultado, saida)
    # {nome:<11} alinha o texto à esquerda em 11 caracteres (fica em coluna).
    print(f"[ok] {nome:<11} -> {saida}")
    return resultado                          # usado pelo mosaico em aplicar_todos


def aplicar_todos(rgb, caminho_entrada):
    """
    Aplica todos os filtros (um arquivo para cada) e também gera
    <imagem>_todos.png: um mosaico com a original e todos os resultados
    lado a lado, com o nome de cada filtro, para facilitar a comparação.
    """
    # Lista de pares (título, imagem). Começa pela original, para comparar.
    paineis = [("original", rgb)]
    for nome in FILTROS:                      # percorre os 6 filtros
        paineis.append((nome, aplicar(rgb, nome, caminho_entrada)))

    # --- Tamanho de cada quadro do mosaico ---
    largura_painel = 500                      # cada quadro tem 500 px de largura
    altura_img, largura_img = rgb.shape[:2]   # [:2] ignora o 3º valor (canais)
    # Altura proporcional, para a imagem não ficar esticada nem achatada.
    altura_painel = round(altura_img * largura_painel / largura_img)
    faixa = 30                                # espaço acima de cada quadro para o título
    colunas = 4                               # 4 quadros por linha
    # Número de linhas arredondado para cima: 7 quadros / 4 colunas = 2 linhas.
    linhas = (len(paineis) + colunas - 1) // colunas

    # Cria uma imagem branca grande o suficiente para todos os quadros.
    mosaico = Image.new("RGB", (colunas * largura_painel,
                                linhas * (altura_painel + faixa)), "white")
    desenho = ImageDraw.Draw(mosaico)         # "caneta" para escrever no mosaico
    try:
        fonte = ImageFont.load_default(size=22)
    except TypeError:                         # Pillow antigo não aceita size
        fonte = ImageFont.load_default()

    # enumerate devolve a posição (i) junto com cada item da lista.
    for i, (nome, matriz) in enumerate(paineis):
        # Converte a matriz em imagem. .convert("RGB") é necessário porque as
        # imagens filtradas são em cinza (1 canal) e o mosaico é colorido.
        img = Image.fromarray(limitar(matriz).astype(np.uint8)).convert("RGB")
        img = img.resize((largura_painel, altura_painel))   # reduz para caber
        # Posição do quadro: i % colunas = coluna (resto da divisão) e
        # i // colunas = linha. Ex.: i=5 -> coluna 1, linha 1.
        x = (i % colunas) * largura_painel
        y = (i // colunas) * (altura_painel + faixa)
        desenho.text((x + 8, y + 4), nome, fill="black", font=fonte)  # título
        mosaico.paste(img, (x, y + faixa))                            # imagem

    saida = caminho_saida(caminho_entrada, "todos")
    mosaico.save(saida)
    print(f"[ok] {'todos':<11} -> {saida}")


def menu(rgb, caminho):
    """Menu interativo: repete até o usuário escolher sair."""
    nomes = list(FILTROS)                     # ["cinza", "blur", "sobel", ...]
    while True:                               # laço infinito; sai com break
        print("\nEscolha uma transformação:")
        # Numera as opções a partir de 1.
        for i, nome in enumerate(nomes, 1):
            print(f"  {i}) {nome}")
        print(f"  {len(nomes) + 1}) todos")   # opção 7
        print("  0) sair")
        # strip tira espaços e lower deixa minúsculo ("Blur " -> "blur").
        opcao = input("> ").strip().lower()
        if opcao in ("0", "sair"):
            break                             # encerra o menu
        # Aceita tanto o número quanto o nome digitado.
        if opcao in ("todos", str(len(nomes) + 1)):
            aplicar_todos(rgb, caminho)
        elif opcao in nomes:
            aplicar(rgb, opcao, caminho)
        elif opcao.isdigit() and 1 <= int(opcao) <= len(nomes):
            # O menu começa em 1, mas a lista começa em 0: por isso o "- 1".
            aplicar(rgb, nomes[int(opcao) - 1], caminho)
        else:
            print("Opção inválida.")


def main():
    """Ponto de entrada: lê os argumentos, carrega a imagem e decide o modo."""
    # sys.argv é a lista do que foi digitado no terminal:
    #   python filtros.py foto.jpg sobel
    #   sys.argv = ["filtros.py", "foto.jpg", "sobel"]
    if len(sys.argv) >= 2:
        caminho = sys.argv[1]                 # o 1º argumento é a imagem
    else:
        # Rodando sem argumentos (ex.: botão "Run" da IDE): pergunta a imagem.
        # .strip('"') tira as aspas que o Windows coloca ao copiar o caminho.
        caminho = input("Caminho da imagem JPG: ").strip().strip('"')

    rgb = carregar_imagem(caminho)
    # shape = (altura, largura, 3); mostramos como largura x altura.
    print(f"Imagem carregada: {caminho} ({rgb.shape[1]}x{rgb.shape[0]})")

    # Sem o nome do filtro -> abre o menu.
    if len(sys.argv) <= 2:
        menu(rgb, caminho)
        return

    # Com o nome do filtro -> aplica direto, sem menu.
    escolha = sys.argv[2].lower()
    if escolha == "todos":
        aplicar_todos(rgb, caminho)
    elif escolha in FILTROS:
        aplicar(rgb, escolha, caminho)
    else:
        print(f"Filtro desconhecido: {escolha}. Opções: {', '.join(FILTROS)}, todos")
        sys.exit(1)                           # 1 = terminou com erro


if __name__ == "__main__":
    main()