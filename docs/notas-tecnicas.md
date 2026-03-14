# 📝 Notas Técnicas do Projeto

Decisões técnicas, limitações encontradas e alternativas documentadas
para referência futura.

---

## Detecção de Acordes e Melodia

### Situação Atual
O servidor de produção roda **Python 3.12**, que é incompatível com
as ferramentas originalmente planejadas:

| Ferramenta | Problema | Substituto Atual |
|---|---|---|
| Basic Pitch (Spotify) | Requer TensorFlow <2.15.1 | librosa chroma/cqt |
| CREPE (MIT) | Requer módulo `imp` removido no Python 3.12 | librosa.pyin |

### Alternativa de Alta Precisão (Implementação Futura)

Caso a qualidade do `librosa` não seja satisfatória, é possível
reativar Basic Pitch e CREPE usando **Python 3.10** — última versão
que suporta TensorFlow <2.15.1.

**Versões compatíveis confirmadas:**

| Componente | Versão |
|---|---|
| Python | 3.10.x |
| TensorFlow | 2.12.0 |
| Basic Pitch | 0.3.3 |
| CREPE | 0.0.13 |

**Como configurar o ambiente alternativo com pyenv:**
```bash
# Instalar Python 3.10 via pyenv
pyenv install 3.10.13
pyenv local 3.10.13

# Recriar o ambiente virtual
rm -rf .venv
python -m venv .venv
source .venv/bin/activate

# Instalar dependências com Basic Pitch e CREPE
pip install basic-pitch==0.3.3
pip install crepe==0.0.13
pip install tensorflow==2.12.0
```

**Quando considerar a migração:**
- A detecção de acordes com librosa errar consistentemente em músicas
  com acordes complexos (jazz, bossa nova, progressões cromáticas)
- A detecção de melodia com librosa.pyin apresentar imprecisões
  perceptíveis na cifra gerada

**Impacto da migração:**
- Requer trocar a versão do Python no servidor
- O CI no GitHub Actions precisará atualizar de `python-version: "3.11"`
  para `python-version: "3.10"`
- Todas as dependências precisarão ser reinstaladas

---

## Separação de Instrumentos

### Modelos Demucs Disponíveis

| Modelo | Faixas | Velocidade | Qualidade | Indicado Para |
|---|---|---|---|---|
| `htdemucs` | 4 (vocals, drums, bass, other) | Mais rápido | Boa | Uso geral |
| `htdemucs_6s` | 6 (+ guitar, piano) | Mais lento | Melhor | Músicas com guitarra ou piano identificáveis |

Configurável via variável de ambiente `DEMUCS_MODEL` no `.env`.