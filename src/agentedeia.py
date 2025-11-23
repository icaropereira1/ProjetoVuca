from crewai import Agent, Task, Crew, Process, LLM

# --- FUNÇÃO 1: ANÁLISE COMPLETA (Para o Botão) ---
def executar_analise_menu(df_dados, api_key, modelo_nome, nome_usuario):
    
    # Prepara os dados
    csv_data = df_dados.to_csv(index=False, sep=';', decimal=',')
    
    # Configura o LLM
    llm = LLM(model=modelo_nome, api_key=api_key)

    # Agentes (Baseados no seu Notebook, mas adaptados)
    analista = Agent(
        role="Analista de Engenharia de Cardápio",
        goal="Classificar itens na Matriz de Engenharia de Menu e achar oportunidades de lucro.",
        backstory="""Você é um especialista em BI de Alimentos e Bebidas. 
        Sua especialidade é cruzar lucratividade vs popularidade.""",
        llm=llm,
        verbose=True
    )

    consultor = Agent(
        role="Consultor de Negócios de Restaurante",
        goal=f"Traduzir a análise técnica para o {nome_usuario} com ações práticas.",
        backstory=f"""Você é um consultor experiente e parceiro do {nome_usuario}. 
        Você fala de forma direta, usa emojis e foca no ROI (Retorno sobre Investimento).""",
        llm=llm,
        verbose=True
    )

    # Tarefas
    t1 = Task(
        description=f"""
        Analise estes dados de vendas:\n{csv_data}
        
        1. Identifique 1 'Quebra-cabeça' (Alta Lucratividade, Baixa Venda).
        2. Identifique 1 'Burro de Carga' (Baixa Lucratividade, Alta Venda).
        """,
        expected_output="Relatório técnico identificando os produtos e os dados deles.",
        agent=analista
    )

    t2 = Task(
        description=f"""
        Escreva uma mensagem de WhatsApp para o {nome_usuario}.
        Dê 2 dicas práticas baseadas nos produtos identificados pelo analista.
        Use tom motivador e emojis.
        """,
        expected_output="Mensagem de WhatsApp pronta.",
        agent=consultor,
        context=[t1]
    )

    crew = Crew(
        agents=[analista, consultor],
        tasks=[t1, t2],
        verbose=True
    )

    return crew.kickoff()

# --- FUNÇÃO 2: CHAT RÁPIDO (Para o Chatbot) ---
def responder_chat_dados(pergunta, df_contexto, api_key, modelo_nome):
    
    csv_contexto = df_contexto.to_csv(index=False, sep=';')
    llm = LLM(model=modelo_nome, api_key=api_key)

    agente_chat = Agent(
        role="Assistente de Dados Financeiros",
        goal="Responder perguntas pontuais sobre o faturamento.",
        backstory="Você é um assistente prestativo com acesso aos dados de vendas.",
        llm=llm,
        verbose=False
    )

    tarefa_chat = Task(
        description=f"""
        Responda à pergunta do usuário com base APENAS nestes dados:
        {csv_contexto}

        PERGUNTA: {pergunta}
        """,
        expected_output="Resposta direta e conversacional.",
        agent=agente_chat
    )

    crew = Crew(agents=[agente_chat], tasks=[tarefa_chat], verbose=False)
    
    return crew.kickoff()