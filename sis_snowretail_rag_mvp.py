# =========================================================
# Snowflake Cortex AI モーニングワークショップ
# 社内問い合わせチャットボットアプリケーション
# =========================================================
# 概要: 
# このアプリケーションは、Snowflake Cortex AIとStreamlit in Snowflakeを使用して、
# 社内問い合わせに対応するチャットボットのデモアプリケーションです。
#
# 機能:
# - シンプルなチャットボット (COMPLETE関数)
# - RAGチャットボット (Cortex Searchを用いた社内文書Q&A)
#
# Created by Tsubasa Kanno @Snowflake
# =========================================================

# =========================================================
# 必要なライブラリのインポート
# =========================================================
# 基本ライブラリ
import streamlit as st
import pandas as pd
import json
import time

# Streamlitの設定
st.set_page_config(layout="wide")

# Snowflake関連ライブラリ
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete as CompleteText
from snowflake.core import Root

# =========================================================
# 定数定義
# =========================================================
# COMPLETE関数用のLLMモデル選択肢
COMPLETE_MODELS = [
    "claude-3-5-sonnet",
    "deepseek-r1",
    "mistral-large2",
    "llama3.3-70b",
    "snowflake-llama-3.3-70b"
]

# =========================================================
# Snowflake接続
# =========================================================

# Snowflakeセッションの取得
snowflake_session = get_active_session()

# =========================================================
# UI関数
# =========================================================

def render_simple_chatbot_page():
    """シンプルチャットボットページを表示します。"""
    st.header("シンプルチャットボット")
    
    # ワークショップ向けの説明
    st.info("""
    ## 🤖 シンプルチャットボットについて
    
    このページでは、Snowflake Cortexの生成AIモデルを使用した基本的なチャットボットを体験できます。
    
    ### 主な機能
    * **テキスト生成**: COMPLETE関数を使用して、入力プロンプトに基づいた応答を生成
    * **チャット履歴の保持**: 会話の文脈を保持し、より自然な対話を実現
    
    ### 大事なポイント
    * このシンプルなチャットボットは外部データを参照せず、モデルの知識だけで応答を生成します
    """)
    
    # セッション状態の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.chat_history = ""
    
    # チャット履歴のクリアボタン
    if st.button("チャット履歴をクリア"):
        st.session_state.messages = []
        st.session_state.chat_history = ""
        st.rerun()
    
    # チャット履歴の表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # ユーザー入力の処理
    if prompt := st.chat_input("メッセージを入力してください"):
        # ユーザーメッセージの表示と履歴の更新
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history += f"User: {prompt}\n"
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            # Cortex Completeを使用して応答を生成
            full_prompt = st.session_state.chat_history + "AI: "
            response = CompleteText(complete_model, full_prompt)
            
            # 応答の表示と履歴の更新
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.session_state.chat_history += f"AI: {response}\n"
            with st.chat_message("assistant"):
                st.markdown(response)
            
        except Exception as e:
            st.error(f"応答の生成中にエラーが発生しました: {str(e)}")

def render_rag_chatbot_page():
    """RAGチャットボットページを表示します。"""
    st.header("社内問い合わせチャットボット")
    
    # ワークショップ向けの説明
    st.info("""
    ## 📚 社内問い合わせチャットボットについて
    
    このページでは、Cortex Searchを用いたRetrieval-Augmented Generation (RAG) フレームワークの高度なチャットボットを体験できます。
    
    ### 主な機能
    * **社内文書の検索**: Cortex Searchを使用して社内文書から関連情報を検索
    * **文脈を考慮した回答生成**: 検索結果を元に、的確な回答を生成
    * **参考文書の表示**: 回答の根拠となった文書を確認可能
    
    ### 大事なポイント
    * 社内文書に関する質問や、製品・サービスに関する具体的な質問をしてみてください
    * 質問が具体的であるほど、より関連性の高いドキュメントが検索されます
    * 参考ドキュメントを展開すると、応答の生成に使用されたドキュメントを確認できます
    * 部署やドキュメントタイプで検索対象を絞り込むことができます
    """)
    
    # Snowflake Root オブジェクトの初期化
    root = Root(snowflake_session)
    
    # 現在のデータベースとスキーマを取得
    current_db_schema = snowflake_session.sql("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()").collect()[0]
    current_database = current_db_schema['CURRENT_DATABASE()']
    current_schema = current_db_schema['CURRENT_SCHEMA()']
    
    # 部署とドキュメントタイプの取得
    try:
        departments = snowflake_session.sql("""
            SELECT DISTINCT department FROM snow_retail_documents
            ORDER BY department
        """).collect()
        department_list = [row['DEPARTMENT'] for row in departments]
        
        document_types = snowflake_session.sql("""
            SELECT DISTINCT document_type FROM snow_retail_documents
            ORDER BY document_type
        """).collect()
        document_type_list = [row['DOCUMENT_TYPE'] for row in document_types]
    except Exception as e:
        st.warning("部署とドキュメントタイプの取得に失敗しました。フィルター機能は使用できません。")
        department_list = []
        document_type_list = []
    
    # 検索フィルターの設定
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
    
    # セッション状態の初期化
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
        st.session_state.rag_chat_history = ""
    
    # チャット履歴のクリアボタン
    if st.button("チャット履歴をクリア"):
        st.session_state.rag_messages = []
        st.session_state.rag_chat_history = ""
        st.rerun()
    
    # チャット履歴の表示
    for message in st.session_state.rag_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "relevant_docs" in message:
                with st.expander("参考ドキュメント"):
                    for doc in message["relevant_docs"]:
                        st.markdown(f"""
                        **タイトル**: {doc['title']}  
                        **種類**: {doc['document_type']}  
                        **部署**: {doc['department']}  
                        **内容**: {doc['content']}
                        """)
    
    # ユーザー入力の処理
    if prompt := st.chat_input("質問を入力してください"):
        # ユーザーメッセージの表示と履歴の更新
        st.session_state.rag_messages.append({"role": "user", "content": prompt})
        st.session_state.rag_chat_history += f"User: {prompt}\n"
        with st.chat_message("user"):
            st.markdown(prompt)
        
        try:
            # Cortex Search Serviceの取得
            try:
                search_service = (
                    root.databases[current_database]
                    .schemas[current_schema]
                    .cortex_search_services["snow_retail_search_service"]
                )
                
                # フィルターの構築
                filter_conditions = []
                
                # 部署フィルターの追加
                if selected_departments:
                    dept_conditions = []
                    for dept in selected_departments:
                        dept_conditions.append({"@eq": {"department": dept}})
                    
                    if len(dept_conditions) == 1:
                        filter_conditions.append(dept_conditions[0])
                    else:
                        filter_conditions.append({"@or": dept_conditions})
                
                # ドキュメントタイプフィルターの追加
                if selected_document_types:
                    type_conditions = []
                    for doc_type in selected_document_types:
                        type_conditions.append({"@eq": {"document_type": doc_type}})
                    
                    if len(type_conditions) == 1:
                        filter_conditions.append(type_conditions[0])
                    else:
                        filter_conditions.append({"@or": type_conditions})
                
                # 最終的なフィルターの組み立て
                search_filter = None
                if filter_conditions:
                    if len(filter_conditions) == 1:
                        search_filter = filter_conditions[0]
                    else:
                        search_filter = {"@and": filter_conditions}
                
                # フィルター情報の表示
                if selected_departments or selected_document_types:
                    filter_info = []
                    if selected_departments:
                        filter_info.append(f"部署: {', '.join(selected_departments)}")
                    if selected_document_types:
                        filter_info.append(f"ドキュメントタイプ: {', '.join(selected_document_types)}")
                    st.info(f"以下の条件で検索します: {' / '.join(filter_info)}")
                
                # 検索の実行（日本語のまま検索）
                search_args = {
                    "query": prompt,
                    "columns": ["title", "chunked_content", "document_type", "department", "document_id"],
                    "limit": 3
                }
                
                # フィルターがある場合は追加
                if search_filter:
                    search_args["filter"] = search_filter
                
                search_results = search_service.search(**search_args)
                
                # 検索結果から元のドキュメントを取得するためのdocument_idリストを取得
                document_ids = [result["document_id"] for result in search_results.results]
                
                # 重複するdocument_idを排除
                unique_document_ids = list(set(document_ids))
                
                # 元のドキュメントテーブルから完全なCONTENTを取得
                original_docs_query = f"""
                    SELECT document_id, title, content, document_type, department
                    FROM snow_retail_documents
                    WHERE document_id IN ({','.join(["'" + str(doc_id) + "'" for doc_id in unique_document_ids])})
                """
                
                original_docs_df = snowflake_session.sql(original_docs_query).collect()
                original_docs = {}
                
                # document_idをキーとした辞書を作成
                for row in original_docs_df:
                    original_docs[row['DOCUMENT_ID']] = {
                        "title": row['TITLE'],
                        "content": row['CONTENT'],
                        "document_type": row['DOCUMENT_TYPE'],
                        "department": row['DEPARTMENT']
                    }
                
                # 検索結果とオリジナルドキュメントを組み合わせて関連ドキュメントリストを作成
                relevant_docs = []
                seen_doc_ids = set()  # 処理済みのドキュメントIDを記録
                
                for result in search_results.results:
                    doc_id = result["document_id"]
                    # 既に処理済みのドキュメントIDはスキップ
                    if doc_id in seen_doc_ids:
                        continue
                        
                    if doc_id in original_docs:
                        relevant_docs.append({
                            "title": original_docs[doc_id]["title"],
                            "content": original_docs[doc_id]["content"],
                            "chunked_content": result["chunked_content"],  # チャンク化されたコンテンツも保持
                            "document_type": original_docs[doc_id]["document_type"],
                            "department": original_docs[doc_id]["department"]
                        })
                        seen_doc_ids.add(doc_id)  # 処理済みとしてマーク
                
                # 検索結果をコンテキストとして使用（チャンク化されたコンテンツを使用）
                context = "参考文書:\n"
                for doc in relevant_docs:
                    context += f"""
                    タイトル: {doc['title']}
                    種類: {doc['document_type']}
                    部署: {doc['department']}
                    内容: {doc['chunked_content']}
                    ---
                    """
                
                # COMPLETEを使用して応答を生成
                prompt_template = f"""
                あなたはスノーリテールの社内アシスタントです。
                以下の文脈を参考に、ユーザーからの質問に日本語で回答してください。
                わからない場合は、その旨を正直に伝えてください。

                文脈:
                {context}

                質問: {prompt}
                """
                
                response = CompleteText(complete_model, prompt_template)
                
                # アシスタントの応答を表示
                with st.chat_message("assistant"):
                    st.markdown(response)
                    with st.expander("参考ドキュメント"):
                        for doc in relevant_docs:
                            st.markdown(f"""
                            **タイトル**: {doc['title']}  
                            **種類**: {doc['document_type']}  
                            **部署**: {doc['department']}  
                            **内容**: {doc['content']}
                            """)
                
                # チャット履歴に追加
                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": response,
                    "relevant_docs": relevant_docs
                })
                st.session_state.rag_chat_history += f"AI: {response}\n"
                
            except Exception as search_error:
                st.error(f"Cortex Search Serviceにアクセスできません。ワークショップでCortex Search Serviceが作成されていることを確認してください。")
                st.code(str(search_error))
                
                # 代わりに通常のCOMPLETE関数で回答を生成
                fallback_response = CompleteText(complete_model, f"以下の質問に日本語で回答してください。社内文書にアクセスできないため、一般的な知識に基づいて回答します。\n\n質問: {prompt}")
                
                with st.chat_message("assistant"):
                    st.markdown(fallback_response)
                    st.info("注: Cortex Search Serviceにアクセスできないため、一般的な知識に基づく回答を生成しています。")
                
                # チャット履歴に追加
                st.session_state.rag_messages.append({
                    "role": "assistant",
                    "content": fallback_response
                })
                st.session_state.rag_chat_history += f"AI: {fallback_response}\n"
            
        except Exception as e:
            st.error(f"応答の生成中にエラーが発生しました: {str(e)}")
            st.code(str(e)) 

# =========================================================
# メイン処理
# =========================================================

# サイドバーでの機能選択
st.sidebar.title("AIモーニングワークショップ")
selected_function = st.sidebar.radio(
    "機能を選択してください",
    ["シンプルチャットボット", "社内問い合わせチャットボット"]
)

# モデル選択（RAGチャットボットで使用）
st.sidebar.title("モデル設定")
complete_model = st.sidebar.selectbox(
    "Completeモデルを選択してください",
    COMPLETE_MODELS,
    index=0
)

# メインコンテンツ
st.title("🏪 スノーリテール 社内問い合わせチャットボット")
st.markdown("---")

# 選択された機能に応じた処理
if selected_function == "シンプルチャットボット":
    render_simple_chatbot_page()
elif selected_function == "社内問い合わせチャットボット":
    render_rag_chatbot_page() 