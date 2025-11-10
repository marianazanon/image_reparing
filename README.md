
## Sobre o Projeto

ADICIONAR AQUI INFO SOBRE O ARTIGO E GRUPO

---

## Requisitos do Sistema

### Requisitos Mínimos

- **Sistema Operacional:** Windows 10+, Linux (Ubuntu 20.04+), ou macOS 10.15+
- **Python:** 3.9 ou superior (recomendado: 3.12)
- **RAM:** 8 GB mínimo (16 GB recomendado)
- **Espaço em Disco:** 2 GB para dependências e modelos
- **GPU (opcional):** NVIDIA CUDA compatível para processamento acelerado

### Dependências Principais

- PyTorch ≥ 2.2.0
- OpenCV ≥ 4.5.0
- NumPy < 2.0.0
- GFPGAN ≥ 1.3.8
- BasicSR 1.4.2

---

## Instalação

### Instalação no Windows

#### 1. Instalar Python

Baixe e instale Python 3.12 de [python.org](https://www.python.org/downloads/)

**Importante:** Durante a instalação, marque a opção "Add Python to PATH"

#### 2. Abrir PowerShell ou Command Prompt

Pressione `Win + R`, digite `cmd` e pressione Enter

#### 3. Navegar até o diretório do projeto

```cmd
cd caminho\para\image_reparing
```

#### 4. Criar ambiente virtual

```cmd
python -m venv .venv
```

#### 5. Ativar ambiente virtual

```cmd
.venv\Scripts\activate
```

Você verá `(.venv)` no início da linha de comando.

#### 6. Instalar dependências

```cmd
pip install --upgrade pip
pip install -r requirements.txt
```

#### 7. Aplicar patch necessário (importante!)

Após a instalação, é necessário corrigir um problema de compatibilidade no pacote `basicsr`:

**Opção A: Comando Automatizado (PowerShell - Recomendado)**

Execute no PowerShell (com ambiente virtual ativado):

```powershell
$patchFile = ".venv\Lib\site-packages\basicsr\data\degradations.py"
(Get-Content $patchFile) -replace 'from torchvision.transforms.functional_tensor import rgb_to_grayscale', 'from torchvision.transforms.functional import rgb_to_grayscale' | Set-Content $patchFile
Write-Host "Patch aplicado com sucesso!" -ForegroundColor Green
```

---

### Instalação no Linux/Mac

#### 1. Verificar instalação do Python

```bash
python3 --version
```

Se não tiver Python 3.9+, instale:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv python3-pip
```

**macOS (com Homebrew):**
```bash
brew install python@3.12
```

#### 2. Navegar até o diretório do projeto

```bash
cd /caminho/para/image_reparing
```

#### 3. Criar ambiente virtual

```bash
python3 -m venv .venv
```

#### 4. Ativar ambiente virtual

```bash
source .venv/bin/activate
```

Você verá `(.venv)` no início da linha de comando.

#### 5. Instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 6. Aplicar patch necessário (importante!)

Execute o seguinte comando para aplicar o patch automaticamente:

```bash
# Localizar o arquivo
PATCH_FILE=".venv/lib/python3.12/site-packages/basicsr/data/degradations.py"

# Aplicar correção
sed -i.bak 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$PATCH_FILE"

echo "Patch aplicado com sucesso!"
```

Ou edite manualmente o arquivo:
```bash
nano .venv/lib/python3.12/site-packages/basicsr/data/degradations.py
```

**Na linha 8, altere:**

De:
```python
from torchvision.transforms.functional_tensor import rgb_to_grayscale
```

Para:
```python
from torchvision.transforms.functional import rgb_to_grayscale
```

Salve com `Ctrl+O`, depois `Enter`, e saia com `Ctrl+X`.

> **Nota:** Este patch precisa ser reaplicado se você reinstalar as dependências.

---

## Configuração dos Modelos

**IMPORTANTE:** Os modelos não estão incluídos no repositório devido ao tamanho (são muito grandes para commit no Git). Você **DEVE** baixá-los manualmente antes de executar o sistema.

O sistema requer modelos pré-treinados que devem ser colocados no diretório `models/`.

### Estrutura Esperada

```
models/
├── GFPGANv1.4.pth                          # Modelo principal GFPGAN (~332 MB) [BAIXAR]
├── deploy.prototxt                          # Configuração do detector de faces [Incluído]
├── res10_300x300_ssd_iter_140000.caffemodel # Pesos do detector (~10 MB) [Incluído]
└── gfpgan/                                  # Pesos auxiliares do GFPGAN
    ├── detection_Resnet50_Final.pth        # Detector de landmarks (~104 MB) [Incluído]
    └── parsing_parsenet.pth                # Parser facial (~81 MB) [Incluído]
```

### Downloads Necessários

#### GFPGANv1.4.pth (OBRIGATÓRIO)

Este arquivo é **muito grande** (~332 MB) e **NÃO** está incluído no repositório.

**Opção 1: Download Direto (Todas as plataformas)**
1. Acesse: [https://github.com/TencentARC/GFPGAN/releases](https://github.com/TencentARC/GFPGAN/releases)
2. Baixe o arquivo `GFPGANv1.4.pth`
3. Mova o arquivo para a pasta `models/` do projeto

**Opção 2: Linha de Comando (Windows - PowerShell)**
```powershell
cd models
Invoke-WebRequest -Uri "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" -OutFile "GFPGANv1.4.pth"
```

**Opção 3: Linha de Comando (Linux/Mac - wget)**
```bash
cd models/
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
```

**Opção 4: Linha de Comando (Linux/Mac - curl)**
```bash
cd models/
curl -L -O https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth
```

**Verificar o download:**

Windows (PowerShell):
```powershell
dir models\GFPGANv1.4.pth
# Deve mostrar: ~332 MB
```

Linux/Mac:
```bash
ls -lh models/GFPGANv1.4.pth
# Deve mostrar: ~332M
```

O arquivo deve ter aproximadamente 332 MB.

#### Outros Modelos

Os demais modelos já estão incluídos no repositório ou são instalados automaticamente com as dependências:

1. **deploy.prototxt** e **res10_300x300_ssd_iter_140000.caffemodel** - Já incluídos
2. **Pesos auxiliares (gfpgan/)** - Já incluídos
3. **Modelos do basicsr/facexlib** - Instalados automaticamente via pip

> **Atenção:** Sem o arquivo `GFPGANv1.4.pth`, o sistema **NÃO FUNCIONARÁ**. Certifique-se de baixá-lo antes de tentar processar imagens.

---

## Como Usar

### Uso Básico

O sistema utiliza o modo **Enhanced** por padrão, que oferece o melhor equilíbrio entre qualidade e velocidade.

#### Windows

```cmd
python -m src.cli --input caminho\para\imagem.jpg --output caminho\para\saida
```

#### Linux/Mac

```bash
python -m src.cli --input caminho/para/imagem.jpg --output caminho/para/saida
```

### Processar uma Única Imagem

```bash
# Modo enhanced (padrão)
python -m src.cli -i data/input/foto.jpg -o data/output

# Com diagnósticos detalhados
python -m src.cli -i data/input/foto.jpg -o data/output --save-diags -v
```

### Processar um Diretório Completo

```bash
# Processar todas as imagens de uma pasta
python -m src.cli -i data/input/ -o data/output/

# Manter estrutura de diretórios
python -m src.cli -i data/input/ -o data/output/ --mirror
```

---

## Presets de Qualidade

O sistema oferece três presets configuráveis via flag `--preset`:

### Standard (Rápido)

**Quando usar:** Testes rápidos, processamento em lote de muitas imagens

```bash
python -m src.cli -i input.jpg -o output/ --preset standard
```

**Características:**
- Upscaling 1x na face (mantém tamanho original)
- Nitidez padrão (0.6x)
- Mesclagem Poisson Mixed
- Tempo: ~0.5-1s por face
- Qualidade: Melhorias moderadas

---

### Enhanced (Padrão - Recomendado)

**Quando usar:** Uso geral, melhor custo-benefício

```bash
python -m src.cli -i input.jpg -o output/
# ou explicitamente:
python -m src.cli -i input.jpg -o output/ --preset enhanced
```

**Características:**
- **Upscaling 2x** na face (dobra a resolução)
- **Nitidez aumentada** (1.2x)
- **Redução de ruído melhorada** (10.0 vs 7.0)
- **Mesclagem Feather** (preserva mais detalhes)
- **Colorização automática** para imagens P&B
- **Qualidade JPEG 98%**
- Tempo: ~1-2s por face
- Qualidade: Melhorias significativas

**Este é o modo padrão!** Oferece resultados excelentes na maioria dos casos.

---

### Maximum (Máxima Qualidade)

**Quando usar:** Fotos importantes, arquivamento, impressão

```bash
python -m src.cli -i input.jpg -o output/ --preset maximum
```

**Características:**
- **Upscaling 4x** na face (quadruplica a resolução)
- **Nitidez máxima** (1.5x)
- **Redução de ruído avançada** (12.0)
- **Mesclagem Feather refinada**
- **Qualidade JPEG 100%** (sem perdas)
- Tempo: ~3-5s por face
- Qualidade: Máxima possível

---

### Comparação de Presets

| Característica | Standard | Enhanced | Maximum |
|----------------|----------|----------|---------|
| Upscaling Face | 1x | 2x * | 4x |
| Nitidez | 0.6 | 1.2 * | 1.5 |
| Redução Ruído | 7.0 | 10.0 * | 12.0 |
| Mesclagem | Poisson | Feather * | Feather |
| Qualidade JPEG | 95% | 98% * | 100% |
| Velocidade | Alta | Média * | Baixa |
| Uso Recomendado | Testes | Uso Geral | Arquivamento |

* = Configuração padrão (Enhanced)

---

## Exemplos de Uso

### 1. Restauração Básica (Modo Enhanced)

```bash
# Imagem única
python -m src.cli -i foto_antiga.jpg -o restaurada/

# Diretório completo
python -m src.cli -i fotos_antigas/ -o restauradas/
```

### 2. Imagem em Preto e Branco (com colorização)

```bash
# Colorização automática (padrão no modo enhanced)
python -m src.cli -i foto_pb.jpg -o restaurada/
```

### 3. Imagem Colorida (sem colorização)

```bash
# Desabilitar colorização para fotos já coloridas
python -m src.cli -i foto_colorida.jpg -o restaurada/ --no-colorize
```

### 4. Máxima Qualidade para Foto Importante

```bash
python -m src.cli \
    -i foto_importante.jpg \
    -o restaurada/ \
    --preset maximum \
    --save-diags \
    -v
```

### 5. Processamento Rápido de Lote

```bash
# Modo standard para processar rapidamente
python -m src.cli \
    -i album_completo/ \
    -o album_restaurado/ \
    --preset standard \
    --mirror
```

### 6. Ajuste Fino de Parâmetros

```bash
# Usar preset enhanced mas com upscaling 4x
python -m src.cli \
    -i foto.jpg \
    -o restaurada/ \
    --preset enhanced \
    --prior-upscale 4

# Personalizar completamente
python -m src.cli \
    -i foto.jpg \
    -o restaurada/ \
    --usm-amount 1.5 \
    --usm-radius 2.0 \
    --prior-upscale 3 \
    --blend-method feather \
    --jpg-quality 100
```

### 7. Detecção Sensível de Faces

```bash
# Reduzir confiança para detectar mais faces
python -m src.cli \
    -i foto_com_faces_pequenas.jpg \
    -o restaurada/ \
    --det-conf 0.4

# Limitar número de faces processadas
python -m src.cli \
    -i foto_grupo.jpg \
    -o restaurada/ \
    --max-faces 3
```

---

## Estrutura do Projeto

```
image_reparing/
├── README.md                    # Este arquivo
├── requirements.txt             # Dependências Python
├── .gitignore                   # Arquivos ignorados pelo Git
│
├── models/                      # Modelos pré-treinados
│   ├── GFPGANv1.4.pth          # Modelo principal (~332 MB)
│   ├── deploy.prototxt         # Config detector de faces
│   ├── res10_300x300_ssd...    # Pesos detector
│   └── gfpgan/                 # Pesos auxiliares GFPGAN
│       ├── detection_Resnet50_Final.pth
│       └── parsing_parsenet.pth
│
├── data/
│   ├── input/                   # Coloque imagens aqui
│   └── output/                  # Imagens restauradas aparecem aqui
│
├── src/                         # Código-fonte
│   ├── __init__.py
│   ├── cli.py                   # Interface de linha de comando
│   │
│   ├── core/                    # Módulos principais
│   │   ├── detector.py          # Detecção de faces (OpenCV DNN)
│   │   ├── align.py             # Alinhamento e crop de faces
│   │   ├── preprocess.py        # Pré-processamento (denoise, sharpen)
│   │   ├── colorizer.py         # Colorização automática
│   │   ├── prior_gran.py        # Restauração com GFPGAN
│   │   ├── blender.py           # Mesclagem de faces
│   │   └── pipeline.py          # Orquestração do pipeline
│   │
│   └── utils/
│       └── image_io.py          # Utilitários de I/O de imagens
│
└── .venv/                       # Ambiente virtual Python (criado na instalação)
```

### Descrição dos Módulos

#### `cli.py` - Interface de Linha de Comando
- Ponto de entrada principal do sistema
- Gerencia argumentos e configurações
- Implementa sistema de presets (standard/enhanced/maximum)
- Coordena processamento em lote

#### `detector.py` - Detecção de Faces
- Utiliza modelo SSD do OpenCV para detectar faces
- Filtragem por confiança e tamanho mínimo
- Non-Maximum Suppression (NMS) para evitar detecções duplicadas

#### `preprocess.py` - Pré-processamento
- **Deblocking:** Remove artefatos de compressão JPEG
- **Denoising:** Redução de ruído com Non-Local Means ou Bilateral
- **Sharpening:** Unsharp Mask para aumentar nitidez
- **Upscaling:** Aumento de resolução (bicúbico ou DNN)

#### `colorizer.py` - Colorização
- Detecta automaticamente imagens em escala de cinza
- Aplica transformação sepia para tons de pele naturais
- Ajusta saturação para cores vibrantes

#### `prior_gran.py` - GFPGAN
- Carrega e executa o modelo GFPGAN
- Restaura detalhes faciais usando prior generativo
- Suporta upscaling 1x, 2x ou 4x

#### `blender.py` - Mesclagem
- **Poisson Blending:** Clonagem sem costura (seamless clone)
- **Feather Blending:** Mesclagem com transição suave
- Máscaras com erosão opcional para evitar halos

#### `pipeline.py` - Orquestração
- Coordena todos os módulos em sequência
- Gerencia múltiplas faces por imagem
- Coleta métricas de tempo de processamento

---

## Resolução de Problemas

### Problema: `ModuleNotFoundError: No module named 'cv2'`

**Causa:** OpenCV não instalado ou ambiente virtual não ativado

**Solução:**
```bash
# Certifique-se que o ambiente está ativado
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Reinstale dependências
pip install -r requirements.txt
```

---

### Problema: `FileNotFoundError: GFPGAN model weights not found`

**Causa:** Modelos não foram baixados ou estão no local errado

**Solução:**
1. Verifique se `models/GFPGANv1.4.pth` existe
2. Baixe os modelos conforme [Configuração dos Modelos](#configuração-dos-modelos)
3. Verifique permissões de leitura dos arquivos

---

### Problema: `ImportError: cannot import name 'rgb_to_grayscale'`

**Causa:** Patch do basicsr não foi aplicado

**Solução:**

**Windows:**
1. Abra o arquivo `.venv\Lib\site-packages\basicsr\data\degradations.py`
2. Na linha 8, altere:
   ```python
   from torchvision.transforms.functional_tensor import rgb_to_grayscale
   ```
   Para:
   ```python
   from torchvision.transforms.functional import rgb_to_grayscale
   ```

**Linux/Mac:**
```bash
sed -i.bak 's/from torchvision.transforms.functional_tensor/from torchvision.transforms.functional/' \
    .venv/lib/python3.12/site-packages/basicsr/data/degradations.py
```

---

### Problema: Faces não são detectadas

**Causa:** Confiança muito alta ou faces muito pequenas

**Solução:**
```bash
# Reduzir threshold de confiança
python -m src.cli -i foto.jpg -o output/ --det-conf 0.4

# Reduzir tamanho mínimo da face
python -m src.cli -i foto.jpg -o output/ --det-min 16
```

---

### Problema: Processamento muito lento

**Causa:** Preset maximum ou hardware limitado

**Solução:**
```bash
# Usar preset standard
python -m src.cli -i foto.jpg -o output/ --preset standard

# Ou reduzir upscaling
python -m src.cli -i foto.jpg -o output/ --prior-upscale 1
```

---

### Problema: Colorização com cores estranhas

**Causa:** Algoritmo simples baseado em sepia

**Solução:**
```bash
# Desabilitar colorização para fotos já coloridas
python -m src.cli -i foto.jpg -o output/ --no-colorize

# Para colorização avançada, use ferramentas especializadas como DeOldify
```

---

### Problema: Erro de memória (CUDA out of memory)

**Causa:** GPU sem memória suficiente

**Solução:**
```bash
# Forçar uso de CPU
python -m src.cli -i foto.jpg -o output/ --device cpu

# Reduzir upscaling
python -m src.cli -i foto.jpg -o output/ --prior-upscale 1
```

---

## Notas Técnicas

### Ambientes Testados

Este projeto foi desenvolvido e testado nos seguintes ambientes:

- **Python:** 3.12.12 (via asdf)
- **Sistema Operacional:** macOS 14+ (compatível com Windows 10+ e Linux Ubuntu 20.04+)
- **Gerenciador de Pacotes:** pip 24.0+

### Versões de Pacotes Principais

```
torch==2.2.2
torchvision==0.17.2
opencv-python==4.9.0
numpy==1.26.4
gfpgan==1.3.8
basicsr==1.4.2 (com patch aplicado)
facexlib==0.3.0
realesrgan==0.3.0
```

### Patch Necessário para basicsr

**Problema de Compatibilidade:**

O pacote `basicsr 1.4.2` possui uma importação incompatível com `torchvision 0.17.x`. É necessário aplicar um patch manual após a instalação.

**Arquivo afetado:**
```
.venv/lib/python3.12/site-packages/basicsr/data/degradations.py
```

**Aplicação automática (Linux/Mac):**
```bash
PATCH_FILE=".venv/lib/python3.12/site-packages/basicsr/data/degradations.py"
sed -i.bak 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$PATCH_FILE"
```

> **Importante:** Este patch deve ser reaplicado sempre que o ambiente virtual for recriado ou as dependências reinstaladas.

### Suporte a GPU (CUDA)

O sistema detecta automaticamente se uma GPU NVIDIA com CUDA está disponível e a utiliza para acelerar o processamento GFPGAN. 

**Requisitos para GPU:**
- NVIDIA GPU com suporte CUDA 11.7+
- Drivers NVIDIA atualizados
- PyTorch compilado com suporte CUDA

**Verificar suporte:**
```python
import torch
print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

**Forçar CPU:**
```bash
python -m src.cli -i foto.jpg -o output/ --device cpu
```

### Tamanho dos Modelos

| Arquivo | Tamanho | Descrição |
|---------|---------|-----------|
| `GFPGANv1.4.pth` | ~332 MB | Modelo principal GFPGAN |
| `res10_300x300_ssd_iter_140000.caffemodel` | ~10 MB | Detector de faces |
| `detection_Resnet50_Final.pth` | ~104 MB | Detector de landmarks |
| `parsing_parsenet.pth` | ~81 MB | Parser facial |
| **Total** | **~527 MB** | Todos os modelos |

### Pipeline de Processamento

O sistema executa as seguintes etapas em ordem:

```
1. Leitura da Imagem
   ↓
2. Colorização (se P&B)
   ↓
3. Detecção de Faces (OpenCV DNN SSD)
   ↓
4. Para cada face:
   ├─ Extração e Padding
   ├─ Pré-processamento
   │  ├─ Deblocking JPEG
   │  ├─ Redução de Ruído
   │  └─ Aumento de Nitidez
   ├─ Restauração GFPGAN
   │  ├─ Detecção de Landmarks
   │  ├─ Reconstrução Neural
   │  └─ Upscaling (1x/2x/4x)
   └─ Mesclagem na Imagem Original
      ├─ Poisson Clone ou
      └─ Feather Blend
   ↓
5. Salvamento da Imagem Restaurada
```

### Métricas de Desempenho

Tempos médios de processamento (Intel i7 / 16GB RAM):

| Preset | Faces/seg | Qualidade | Uso de Memória |
|--------|-----------|-----------|----------------|
| Standard | ~1-2 | Boa | ~2 GB |
| Enhanced | ~0.5-1 | Ótima | ~3 GB |
| Maximum | ~0.2-0.3 | Excelente | ~4 GB |

*Tempos aproximados para faces de 512x512 pixels*

### Limitações Conhecidas

1. **Colorização Simples:** O algoritmo de colorização usa transformação sepia. Para resultados artísticos, use ferramentas especializadas como DeOldify.

2. **Detecção de Faces:** Funciona melhor com faces frontais. Perfis extremos ou oclusões podem não ser detectados.

3. **Memória GPU:** Upscaling 4x requer GPU com pelo menos 6GB VRAM.

4. **Formato de Saída:** Salva em JPEG por padrão. Para imagens sem perda, use `--jpg-quality 100` ou converta para PNG posteriormente.

---

## Comandos Úteis

### Ver Todos os Parâmetros Disponíveis

```bash
python -m src.cli --help
```

### Ativar/Desativar Ambiente Virtual

**Linux/Mac:**
```bash
# Ativar
source .venv/bin/activate

# Desativar
deactivate
```

**Windows:**
```cmd
REM Ativar
.venv\Scripts\activate

REM Desativar
deactivate
```

### Verificar Instalação

```bash
# Verificar Python
python --version

# Verificar pacotes instalados
pip list

# Testar importação
python -c "import torch, cv2, numpy; print('OK')"
```

### Limpar Cache Python

```bash
# Linux/Mac
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Windows PowerShell
Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Force -Recurse
Get-ChildItem -Path . -Include *.pyc -Recurse -Force | Remove-Item -Force
```
---

## Referências

### Artigos Científicos

1. **GFPGAN:** Wang, X., Li, Y., Zhang, H., & Shan, Y. (2021). "Towards Real-World Blind Face Restoration with Generative Facial Prior." CVPR 2021.

2. **Face Detection:** Zhang, K., Zhang, Z., Li, Z., & Qiao, Y. (2016). "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks."

### Repositórios

- [GFPGAN Official Repository](https://github.com/TencentARC/GFPGAN)
- [OpenCV DNN Models](https://github.com/opencv/opencv)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
