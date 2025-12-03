import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv

# --- IMPORTAÇÃO DA LÓGICA DE IA (src/agentedeia.py) ---
# Certifique-se de que o arquivo agentedeia.py esteja dentro da pasta 'src'
try:
    from src.agentedeia import executar_analise_menu, responder_chat_dados
except ImportError:
    st.error("Erro ao importar 'src.agentedeia'. Verifique se o arquivo existe e se a estrutura de pastas está correta.")
    st.stop()

# --- CONFIGURAÇÃO INICIAL ---
load_dotenv()
st.set_page_config(page_title="ChefIA - Inteligência de Menu", layout="wide", page_icon="👨‍🍳")

# --- FUNÇÕES AUXILIARES ---
def classificar_produto(row, pop, luc):
    if row['popularidade'] >= pop and row['lucratividade'] >= luc: return '⭐ Estrela'
    elif row['popularidade'] >= pop and row['lucratividade'] < luc: return '🛒 Popular'
    elif row['popularidade'] < pop and row['lucratividade'] >= luc: return '💎 Oportunidade'
    else: return '⚠️ Crítico'

def limpar_texto_ia(texto_obj):
    # Garante que o output seja string pura para evitar erros de renderização
    texto = str(texto_obj.raw) if hasattr(texto_obj, 'raw') else str(texto_obj)
    return texto.replace("$", "\\$")

CORES_MATRIZ = {
    '⭐ Estrela': '#FFD700',
    '🛒 Popular': '#1E90FF',
    '💎 Oportunidade': '#32CD32',
    '⚠️ Crítico': '#FF4500'
}

# --- INTERFACE LATERAL ---
st.sidebar.title("🔧 Configurações ChefIA")
provedor = st.sidebar.selectbox("Selecione a LLM:", ["Gemini", "OpenAI (ChatGPT)", "DeepSeek", "Perplexity"])

api_key_final = None
modelo_selecionado = None

# Configuração das chaves e strings de modelo compatíveis com litellm
if provedor == "Gemini":
    mod = st.sidebar.selectbox("Modelo:", ["gemini-2.0-flash","gemini-2.0-flash-lite","gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro","gemini-3-pro-preview"])
    # O prefixo 'gemini/' é crucial para o litellm identificar o provedor Google
    modelo_selecionado = f"gemini/{mod}"
    api_key_final = os.getenv("GOOGLE_API_KEY") or st.sidebar.text_input("Google API Key:", type="password")

elif provedor == "OpenAI (ChatGPT)":
    mod = st.sidebar.selectbox("Modelo:", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"])
    modelo_selecionado = f"openai/{mod}"
    api_key_final = os.getenv("OPENAI_API_KEY") or st.sidebar.text_input("OpenAI API Key:", type="password")

elif provedor == "DeepSeek":
    mod = st.sidebar.selectbox("Modelo:", ["deepseek-chat", "deepseek-coder"])
    modelo_selecionado = f"deepseek/{mod}"
    api_key_final = os.getenv("DEEPSEEK_API_KEY") or st.sidebar.text_input("DeepSeek API Key:", type="password")

elif provedor == "Perplexity":
    mod = st.sidebar.selectbox("Modelo:", ["sonar-pro", "sonar", "sonar-reasoning"])
    modelo_selecionado = f"perplexity/{mod}"
    api_key_final = os.getenv("PERPLEXITY_API_KEY") or st.sidebar.text_input("Perplexity API Key:", type="password")

# --- CABEÇALHO E NOME ---
st.title("👨‍🍳 ChefIA - Inteligência Gastronômica")

if 'user_name' not in st.session_state: st.session_state.user_name = ''
if st.session_state.user_name == '':
    st.info("Olá! Para começarmos, como gostaria de ser chamado?")
    if n := st.text_input("Seu nome:"):
        st.session_state.user_name = n
        st.rerun()
    st.stop()

st.markdown(f"Painel de Controle de **{st.session_state.user_name}**")

# ==============================================================================
# SEÇÃO 1: SOBRE O PROJETO (EXPANDER)
# ==============================================================================
with st.expander("💡 Entenda como o ChefIA funciona (Clique para abrir)", expanded=False):
    st.markdown("""
    ### 🎯 O Objetivo
    O **ChefIA** cruza seus dados de **Vendas (Popularidade)** com suas **Fichas Técnicas (Lucratividade)** para criar uma Matriz de Engenharia de Menu.
    
    ### 🧩 Classificação dos Pratos
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        st.success("**⭐ Estrela (Alta Venda / Alto Lucro)**\n\nSão os campeões. Mantenha a qualidade e destaque no cardápio.")
        st.info("**🛒 Popular (Alta Venda / Baixo Lucro)**\n\nTrazem fluxo, mas pouca margem. Tente reduzir custo ou aumentar levemente o preço.")
    with c2:
        st.warning("**💎 Oportunidade (Baixa Venda / Alto Lucro)**\n\nLucrativos, mas ninguém pede. Faça promoções e fotos melhores.")
        st.error("**⚠️ Crítico (Baixa Venda / Baixo Lucro)**\n\nNão vendem e não dão lucro. Considere remover do cardápio.")

st.markdown("---")

# ==============================================================================
# SEÇÃO 2: GESTÃO DE DADOS (INPUT MANUAL E IMPORTAÇÃO)
# ==============================================================================
st.header("📝 Gerenciamento do Cardápio")
st.markdown("Insira seus dados reais aqui. Você pode importar um arquivo ou digitar manualmente.")

if 'dados_manuais' not in st.session_state:
    st.session_state.dados_manuais = []

# --- IMPORTAÇÃO ---
with st.expander("📂 Importar Arquivo CSV (Backup)", expanded=False):
    up_sim = st.file_uploader("Escolha o arquivo CSV", type=['csv'])
    
    if up_sim is not None:
        file_id = f"{up_sim.name}_{up_sim.size}"
        if 'ultimo_import_id' not in st.session_state or st.session_state.ultimo_import_id != file_id:
            try:
                df_import = pd.read_csv(up_sim, sep=';', decimal=',')
                df_import.columns = df_import.columns.str.strip().str.lower()
                
                cols_map = {
                    "produto_nome": "produto_nome", 
                    "custo_producao": "custo_producao", 
                    "preco_venda": "preco_venda", 
                    "popularidade": "popularidade"
                }
                
                if all(c in df_import.columns for c in cols_map.keys()):
                    df_import['popularidade'] = pd.to_numeric(df_import['popularidade'], errors='coerce').fillna(0).astype(int)
                    df_import['custo_producao'] = pd.to_numeric(df_import['custo_producao'], errors='coerce').fillna(0.0)
                    df_import['preco_venda'] = pd.to_numeric(df_import['preco_venda'], errors='coerce').fillna(0.0)
                    
                    st.session_state.dados_manuais = df_import.to_dict('records')
                    st.session_state.ultimo_import_id = file_id
                    st.success("Dados carregados com sucesso!")
                    st.rerun()
                else:
                    st.error("O arquivo CSV deve conter as colunas: produto_nome, custo_producao, preco_venda, popularidade.")
            except Exception as e:
                st.error(f"Erro ao ler arquivo: {e}")

# --- ADIÇÃO MANUAL ---
with st.expander("➕ Adicionar Prato Novo (Formulário)", expanded=False):
    with st.form("form_manual"):
        c1, c2, c3, c4 = st.columns(4)
        novo_nome = c1.text_input("Nome do Prato")
        novo_custo = c2.number_input("Custo (R$)", min_value=0.01, format="%.2f")
        novo_preco = c3.number_input("Preço Venda (R$)", min_value=0.01, format="%.2f")
        novo_qtd = c4.number_input("Vendas (Qtd)", min_value=1, step=1)
        
        if st.form_submit_button("Adicionar"):
            if novo_nome:
                st.session_state.dados_manuais.append({
                    "produto_nome": novo_nome.upper(),
                    "custo_producao": novo_custo,
                    "preco_venda": novo_preco,
                    "popularidade": int(novo_qtd)
                })
                st.rerun()

# --- TABELA EDITÁVEL ---
st.markdown("### 📋 Seus Dados")
st.info("💡 Dica: Clique em qualquer célula para editar os valores.")

if len(st.session_state.dados_manuais) > 0:
    df_input = pd.DataFrame(st.session_state.dados_manuais)
    cols_keep = ["produto_nome", "custo_producao", "preco_venda", "popularidade"]
    for c in cols_keep:
        if c not in df_input.columns: df_input[c] = 0
    df_input = df_input[cols_keep]
else:
    df_input = pd.DataFrame(columns=["produto_nome", "custo_producao", "preco_venda", "popularidade"])

column_cfg = {
    "produto_nome": st.column_config.TextColumn("Nome", required=True),
    "custo_producao": st.column_config.NumberColumn("Custo", min_value=0.01, format="R$ %.2f", required=True),
    "preco_venda": st.column_config.NumberColumn("Venda", min_value=0.01, format="R$ %.2f", required=True),
    "popularidade": st.column_config.NumberColumn("Qtd", min_value=1, step=1, required=True)
}

edited_df = st.data_editor(
    df_input,
    column_config=column_cfg,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    key="editor_dados"
)

if not edited_df.equals(df_input):
    st.session_state.dados_manuais = edited_df.to_dict('records')

# ==============================================================================
# SEÇÃO 3: ANÁLISE E INTELIGÊNCIA (DASHBOARD)
# ==============================================================================

if not edited_df.empty:
    df_final = edited_df.copy()
    df_final['lucratividade'] = df_final['preco_venda'] - df_final['custo_producao']
    df_final['receita_total'] = df_final['preco_venda'] * df_final['popularidade']

    st.markdown("---")
    st.header("📊 Dashboard & Inteligência")

    # Botões de Ação de Dados
    c_b1, c_b2 = st.columns([1, 1])
    with c_b1:
        if st.button("🗑️ Limpar Todos os Dados"):
            st.session_state.dados_manuais = []
            st.rerun()
    with c_b2:
        csv = df_final.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
        st.download_button("💾 Baixar Backup dos Dados", data=csv, file_name='dados_chefia.csv', mime='text/csv')

    if len(df_final) >= 1:
        ref_pop = df_final['popularidade'].mean()
        ref_luc = df_final['lucratividade'].mean()
        
        df_final['classificacao'] = df_final.apply(lambda x: classificar_produto(x, ref_pop, ref_luc), axis=1)

        # KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Itens", len(df_final))
        k2.metric("⭐ Estrelas", len(df_final[df_final['classificacao']=='⭐ Estrela']))
        k3.metric("💎 Oportunidades", len(df_final[df_final['classificacao']=='💎 Oportunidade']))
        k4.metric("⚠️ Críticos", len(df_final[df_final['classificacao']=='⚠️ Crítico']))
        k5.metric("🛒 Populares", len(df_final[df_final['classificacao']=='🛒 Popular']))

        # Gráfico
        fig_sim = px.scatter(
            df_final, x="popularidade", y="lucratividade", color="classificacao",
            size="popularidade", hover_name="produto_nome", text="produto_nome",
            color_discrete_map=CORES_MATRIZ, template="plotly_white", title="Matriz de Engenharia de Menu"
        )
        fig_sim.add_vline(x=ref_pop, line_dash="dash", line_color="gray", annotation_text="Média Pop.")
        fig_sim.add_hline(y=ref_luc, line_dash="dash", line_color="gray", annotation_text="Média Lucro")
        fig_sim.update_traces(textposition='top center')
        st.plotly_chart(fig_sim, use_container_width=True)

        # --- ABAS DE INTELIGÊNCIA ---
        st.markdown("### 🧠 Inteligência Artificial")
        
        tab1, tab2 = st.tabs(["📋 Relatório Estratégico", "💬 Perguntar aos Dados"])
        
        # ABA 1: Relatório
        with tab1:
            st.info(f"Modelo selecionado: **{modelo_selecionado}**")
            if st.button("💡 Gerar Relatório Automático"):
                if not api_key_final:
                    st.error("⚠️ Configure a API Key na barra lateral para usar a IA.")
                else:
                    with st.spinner(f"Engenheiro de Menu e Consultor trabalhando..."):
                        try:
                            # Pega extremos para análise (foca no que importa para economizar tokens)
                            df_analise = pd.concat([
                                df_final.sort_values('lucratividade', ascending=False).head(10),
                                df_final.sort_values('popularidade', ascending=False).head(10),
                                df_final.sort_values('lucratividade', ascending=True).head(5)
                            ]).drop_duplicates()
                            
                            # CHAMADA DA NOVA FUNÇÃO DO ARQUIVO EXTERNO
                            # Note a ordem dos argumentos definida em agentedeia.py: (dados, api_key, modelo)
                            res = executar_analise_menu(df_analise, api_key_final, modelo_selecionado)
                            
                            st.markdown(limpar_texto_ia(res))
                        except Exception as e:
                            st.error(f"Erro na IA: {e}")

        # ABA 2: Chatbot
        with tab2:
            st.write("Faça perguntas livres sobre seus dados (faturamento, custos, margens).")
            
            if "messages" not in st.session_state:
                st.session_state.messages = []

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ex: Qual o produto com maior faturamento total?"):
                if not api_key_final:
                    st.error("⚠️ Configure a API Key na barra lateral.")
                else:
                    st.chat_message("user").markdown(prompt)
                    st.session_state.messages.append({"role": "user", "content": prompt})

                    with st.chat_message("assistant", avatar="👨‍🍳"):
                        with st.spinner("Calculando..."):
                            try:
                                # Passa os dados para o chat
                                df_contexto = df_final.sort_values(by='receita_total', ascending=False).head(60)
                                
                                # CHAMADA DA NOVA FUNÇÃO DO ARQUIVO EXTERNO
                                # Note a ordem dos argumentos: (pergunta, dados, api_key, modelo)
                                resposta_raw = responder_chat_dados(prompt, df_contexto, api_key_final, modelo_selecionado)
                                
                                resposta = limpar_texto_ia(resposta_raw)
                                
                                st.markdown(resposta)
                                st.session_state.messages.append({"role": "assistant", "content": resposta})
                            except Exception as e:
                                st.error(f"Erro ao responder: {e}")
else:
    st.info("👆 Adicione pratos manualmente ou importe um CSV para começar a análise.")