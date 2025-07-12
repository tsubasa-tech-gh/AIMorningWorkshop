# =========================================================
# Snowflake AI実践ワークショップ
# 社内問い合わせチャットボットアプリケーション
# =========================================================
# 概要: 
# このアプリケーションは、Snowflake Cortex AIとStreamlit in Snowflakeを使用して、
# 社内問い合わせに対応するチャットボットのデモアプリケーションです。
#
# 機能:
# - シンプルなチャットボット (AI_COMPLETE関数)
# - RAGチャットボット (Cortex Searchを用いた社内文書Q&A)
#
# Created by Tsubasa Kanno @Snowflake
# =========================================================

import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.core import Root

# Streamlitの設定
st.set_page_config(layout="wide")

# =========================================================
# 定数定義
# =========================================================
# AI_COMPLETE関数用のLLMモデル選択肢
AI_COMPLETE_MODELS = [
    "llama4-maverick",
    "claude-3-5-sonnet",
    "mistral-large2"
]

# Snowflakeセッションの取得
snowflake_session = get_active_session()

# =========================================================
# ユーティリティ関数
# =========================================================
def get_ai_response(model: str, prompt: str):
    """AI応答を取得してエスケープ処理を行う"""
    try:
        escaped_prompt = prompt.replace("'", "''")
        
        response_query = f"""
        SELECT AI_COMPLETE(
            '{model}',
            '{escaped_prompt}'
        ) as response
        """
        
        response_result = snowflake_session.sql(response_query).collect()
        
        if response_result and response_result[0]['RESPONSE']:
            response = response_result[0]['RESPONSE']
            
            # 応答の後処理
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]
            
            response = response.replace('\\n', '\n')
            response = response.replace('\\t', '\t')
            response = response.replace('\\"', '"')
            response = response.replace("\\'", "'")
            response = response.replace('\\\\', '\\')
            
            return response
        else:
            return "応答を取得できませんでした。"
            
    except Exception as e:
        return f"エラーが発生しました: {str(e)}"

@st.fragment
def render_document_details(doc, doc_index, unique_id):
    """ドキュメントの詳細表示を管理するフラグメント"""
    toggle_key = f"show_details_{unique_id}_doc_{doc_index}"
    if toggle_key not in st.session_state:
        st.session_state[toggle_key] = False
    
    st.markdown(f"**📋 タイトル**: {doc['title']}")
    st.markdown(f"**📂 種類**: {doc['document_type']} | **🏢 部署**: {doc['department']}")
    
    content = doc['content']
    max_chars = 200
    
    if len(content) <= max_chars:
        st.markdown(f"**📖 内容**: {content}")
    else:
        if st.session_state[toggle_key]:
            st.markdown(f"**📖 内容**: {content}")
            if st.button("🔼 概要のみ表示", key=f"hide_{toggle_key}"):
                st.session_state[toggle_key] = False
                st.rerun(scope="fragment")
        else:
            st.markdown(f"**📖 内容**: {content[:max_chars]}...")
            if st.button("🔽 詳細を表示", key=f"show_{toggle_key}"):
                st.session_state[toggle_key] = True
                st.rerun(scope="fragment")

# =========================================================
# UI関数
# =========================================================

def render_simple_chatbot_page():
    """シンプルチャットボットページを表示"""
    st.header("シンプルチャットボット")
    
    st.info("""
    ## 🤖 シンプルチャットボットについて
    
    このページでは、Snowflake CortexのAI_COMPLETE関数を使用した基本的なチャットボットを体験できます。
    
    ### 主な機能
    * **テキスト生成**: AI_COMPLETE関数を使用して、入力プロンプトに基づいた応答を生成
    * **チャット履歴の保持**: 会話の文脈を保持し、より自然な対話を実現
    
    ### 重要なポイント
    * このチャットボットは外部データを参照せず、モデルの知識だけで応答を生成します
    """)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if st.button("チャット履歴をクリア"):
        st.session_state.messages = []
        st.rerun()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("メッセージを入力してください"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            chat_history = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in st.session_state.messages[-5:]
            ])
            
            response = get_ai_response(ai_complete_model, chat_history + "\nAI: ")
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
                
        except Exception as e:
            st.error(f"応答の生成中にエラーが発生しました: {str(e)}")

def render_rag_chatbot_page():
    """RAGチャットボットページを表示"""
    st.header("社内問い合わせチャットボット")
    
    st.info("""
    ## 📚 社内問い合わせチャットボットについて
    
    このページでは、Cortex Searchを用いたRetrieval-Augmented Generation (RAG) フレームワークの高度なチャットボットを体験できます。
    
    ### 主な機能
    * **社内文書の検索**: Cortex Searchを使用して社内文書から関連情報を検索
    * **文脈を考慮した回答生成**: 検索結果を元に、的確な回答を生成
    * **参考文書の表示**: 回答の根拠となった文書を確認可能
    
    ### 使用方法
    * 社内文書に関する質問や、製品・サービスに関する具体的な質問をしてください
    * 部署やドキュメントタイプで検索対象を絞り込むことができます
    """)
    
    root = Root(snowflake_session)
    
    current_info = snowflake_session.sql("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()").collect()[0]
    current_database = current_info['CURRENT_DATABASE()']
    current_schema = current_info['CURRENT_SCHEMA()']
    
    try:
        departments = snowflake_session.sql(
            "SELECT DISTINCT department FROM snow_retail_documents ORDER BY department"
        ).collect()
        department_list = [row['DEPARTMENT'] for row in departments]
        
        document_types = snowflake_session.sql(
            "SELECT DISTINCT document_type FROM snow_retail_documents ORDER BY document_type"
        ).collect()
        document_type_list = [row['DOCUMENT_TYPE'] for row in document_types]
    except Exception as e:
        st.warning("フィルター情報の取得に失敗しました。基本機能のみ利用可能です。")
        department_list = []
        document_type_list = []
    
    with st.expander("検索フィルター設定", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            selected_departments = st.multiselect(
                "部署で絞り込み",
                options=department_list,
                default=[]
            )
        
        with col2:
            selected_document_types = st.multiselect(
                "ドキュメントタイプで絞り込み",
                options=document_type_list,
                default=[]
            )
    
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    
    if st.button("チャット履歴をクリア"):
        st.session_state.rag_messages = []
        st.rerun()
    
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "relevant_docs" in message:
                with st.expander("参考ドキュメント"):
                    for i, doc in enumerate(message["relevant_docs"]):
                        render_document_details(doc, i, hash(str(message['content'])))
                        if i < len(message["relevant_docs"]) - 1:
                            st.markdown("---")
    
    if prompt := st.chat_input("質問を入力してください"):
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            search_service = (
                root.databases[current_database]
                .schemas[current_schema]
                .cortex_search_services["snow_retail_search_service"]
            )
            search_filter = None
            if selected_departments or selected_document_types:
                filter_conditions = []
                
                if selected_departments:
                    dept_conditions = [{"@eq": {"department": dept}} for dept in selected_departments]
                    filter_conditions.append({"@or": dept_conditions} if len(dept_conditions) > 1 else dept_conditions[0])
                
                if selected_document_types:
                    type_conditions = [{"@eq": {"document_type": doc_type}} for doc_type in selected_document_types]
                    filter_conditions.append({"@or": type_conditions} if len(type_conditions) > 1 else type_conditions[0])
                
                search_filter = {"@and": filter_conditions} if len(filter_conditions) > 1 else filter_conditions[0]
            search_args = {
                "query": prompt,
                "columns": ["title", "chunked_content", "document_type", "department", "document_id"],
                "limit": 3
            }
            
            if search_filter:
                search_args["filter"] = search_filter
            
            search_results = search_service.search(**search_args)
            
            if search_results.results:
                document_ids = list(set([result["document_id"] for result in search_results.results]))
                
                original_docs_query = f"""
                SELECT document_id, title, content, document_type, department
                FROM snow_retail_documents
                WHERE document_id IN ({','.join(["'" + str(doc_id) + "'" for doc_id in document_ids])})
                """
                
                original_docs_df = snowflake_session.sql(original_docs_query).collect()
                original_docs = {row['DOCUMENT_ID']: row for row in original_docs_df}
                
                relevant_docs = []
                seen_doc_ids = set()
                
                for result in search_results.results:
                    doc_id = result["document_id"]
                    if doc_id not in seen_doc_ids and doc_id in original_docs:
                        doc_info = original_docs[doc_id]
                        relevant_docs.append({
                            "title": doc_info["TITLE"],
                            "content": doc_info["CONTENT"],
                            "document_type": doc_info["DOCUMENT_TYPE"],
                            "department": doc_info["DEPARTMENT"]
                        })
                        seen_doc_ids.add(doc_id)
                
                context = "参考文書:\n"
                for doc in relevant_docs:
                    context += f"""
                    タイトル: {doc['title']}
                    種類: {doc['document_type']}
                    部署: {doc['department']}
                    内容: {doc['content']}
                    ---
                    """
                
                prompt_template = f"""
                あなたはスノーリテールの社内アシスタントです。
                以下の文脈を参考に、ユーザーからの質問に日本語で回答してください。
                わからない場合は、その旨を正直に伝えてください。

                文脈:
                {context}

                質問: {prompt}
                """
                
                response = get_ai_response(ai_complete_model, prompt_template)
                with st.chat_message("assistant"):
                    st.markdown(response)
                    with st.expander("参考ドキュメント"):
                        for i, doc in enumerate(relevant_docs):
                            render_document_details(doc, i, hash(str(prompt)))
                            if i < len(relevant_docs) - 1:
                                st.markdown("---")
                
                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": response,
                    "relevant_docs": relevant_docs
                })
            else:
                st.warning("関連するドキュメントが見つかりませんでした。")
                
        except Exception as e:
            st.error(f"検索または応答の生成中にエラーが発生しました: {str(e)}")
            
            try:
                fallback_prompt = f"以下の質問に日本語で回答してください。社内文書にアクセスできないため、一般的な知識に基づいて回答します。\n\n質問: {prompt}"
                
                fallback_response = get_ai_response(ai_complete_model, fallback_prompt)
                
                with st.chat_message("assistant"):
                    st.markdown(fallback_response)
                    st.info("注: Cortex Search Serviceにアクセスできないため、一般的な知識に基づく回答を生成しています。")
                
                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": fallback_response
                })
                
            except Exception as fallback_error:
                st.error(f"フォールバック処理でもエラーが発生しました: {str(fallback_error)}")

# =========================================================
# メイン処理
# =========================================================

# サイドバー設定
st.sidebar.title("AIモーニングワークショップ")
selected_function = st.sidebar.radio(
    "機能を選択してください",
    ["シンプルチャットボット", "社内問い合わせチャットボット"]
)

# モデル選択
st.sidebar.title("モデル設定")
ai_complete_model = st.sidebar.selectbox(
    "AI_COMPLETEモデルを選択してください",
    AI_COMPLETE_MODELS,
    index=0
)

# メインコンテンツ
st.title("🏪 スノーリテール 社内問い合わせチャットボット")
st.markdown("---")

# 選択された機能の実行
if selected_function == "シンプルチャットボット":
    render_simple_chatbot_page()
elif selected_function == "社内問い合わせチャットボット":
    render_rag_chatbot_page() 
