# 🎸 MelodyTab AI

> Gerador automático de cifras melódicas com Inteligência Artificial

![CI](https://github.com/seu-usuario/melodytab-ai/actions/workflows/ci.yml/badge.svg?branch=master)

MelodyTab AI recebe um arquivo de áudio musical e gera automaticamente
uma cifra melódica completa — com acordes, linha melódica e letra
sincronizados — utilizando exclusivamente ferramentas gratuitas de IA.

---

## 🎵 Exemplo de Saída
```
[C] Quando o sol se [Am] pôr
   ré  mi  sol      mi ré dó
[F] Estarei aqui pra [G7] te esperar
```

---

## ⚙️ Como Instalar

**1. Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/melodytab-ai.git
cd melodytab-ai
```

**2. Crie o ambiente virtual e instale as dependências:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Configure as variáveis de ambiente:**
```bash
cp .env.example .env
# Edite o .env com suas chaves de API
code .env
```

---

## ▶️ Como Usar
```bash
source .venv/bin/activate
streamlit run app/main.py
```

Acesse `http://localhost:8501` no navegador, faça upload de um arquivo
MP3, WAV ou FLAC e clique em **Gerar Cifra**.

---

## 🔑 Variáveis de Ambiente

Veja o arquivo [.env.example](.env.example) para a lista completa
de variáveis necessárias e onde obter cada chave de API.

---

## 🏗️ Arquitetura

Veja [docs/architecture.md](docs/architecture.md) para o detalhamento
completo do pipeline de processamento e decisões de arquitetura.

---

## 🤝 Como Contribuir

Veja [docs/contributing.md](docs/contributing.md) para o guia completo
de contribuição, padrão de commits e fluxo de branches.

---

## 📄 Licença

MIT License — veja o arquivo [LICENSE](LICENSE) para detalhes.