import streamlit as st
import os
import chromadb
from groq import Groq
from dotenv import load_dotenv
from pypdf import PdfReader

# --- 1. 초기 설정 및 페이지 레이아웃 ---
load_dotenv()
st.set_page_config(page_title="RAG AI Agent", page_icon="🧬", layout="wide")
st.title("RAG AI Agent")
st.sidebar.header("설정 및 동기화")

# Groq 클라이언트
if "groq_client" not in st.session_state:
    st.session_state.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ChromaDB 설정
@st.cache_resource
def get_chroma_client():
    client = chromadb.PersistentClient(path="./juhyeong_advanced_db")
    return client.get_or_create_collection(name="advanced_tech_notes")

collection = get_chroma_client()

# --- 2. 기능부: 문서 로드 및 청킹 (성능 개선) ---
def load_files_to_db(directory="./data"):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    chunk_size = 800  # 검색 정밀도를 위해 청크 크기를 소폭 조정
    overlap = 150     # 문맥 연결을 위해 겹침 구간 증설

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        content = ""
        
        if filename.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                st.sidebar.error(f"{filename} 읽기 실패: {e}")
        elif filename.endswith(".pdf"):
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text = page.extract_text()
                    if text: content += text + "\n"
            except Exception as e:
                st.sidebar.error(f"{filename} PDF 읽기 실패: {e}")
        
        if content:
            # 텍스트를 조각내어 저장
            for i in range(0, len(content), chunk_size - overlap):
                chunk = content[i : i + chunk_size]
                chunk_id = f"{filename}_part_{i}"
                collection.upsert(
                    documents=[chunk],
                    metadatas=[{"source": filename}],
                    ids=[chunk_id]
                )
    return "✨ 모든 로컬 문서(PDF/TXT) 동기화 완료!"

# 사이드바 동기화 버튼
if st.sidebar.button("📂 데이터 폴더와 동기화"):
    with st.spinner("데이터베이스를 업데이트 중..."):
        msg = load_files_to_db()
        st.sidebar.success(msg)

# --- 3. 채팅 인터페이스 구현 ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력
if query := st.chat_input("질문을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # RAG 엔진 가동
    with st.chat_message("assistant"):
        with st.spinner("문서에서 관련 내용을 찾는 중..."):
            # 1. ChromaDB 검색 (n_results=3 최적값 적용)
            results = collection.query(query_texts=[query], n_results=3)
            
            contexts = results['documents'][0]
            metadatas = results['metadatas'][0]
            
            # 2. 참고 문헌 텍스트 구성 (파일명과 내용을 명확히 매칭)
            # study_rag_3.py의 방식을 채택하여 AI가 어떤 파일의 내용인지 정확히 알게 함
            formatted_contexts = []
            for doc, meta in zip(contexts, metadatas):
                formatted_contexts.append(f"[{meta['source']}]: {doc}")
            
            context_text = "\n\n".join(formatted_contexts)
            sources = list(set([m['source'] for m in metadatas]))

            # 3. 프롬프트 개선 (개인화된 지시사항 반영)
            prompt = f"""
            너는 전문 지식 비서야. [참고 문헌]을 바탕으로 질문에 답해줘.
            IT 지식을 모두 활용하고, 반드시 출처를 밝혀줘.
            그리고 너의 의견으로 하지말고 반드시 data폴더 안의 txt, pdf 파일 기반으로 답변을 해줘
            그리고 반드시 출처를 밝혀줘 어떤 txt, pdf 파일을 참고했는지를 말해줘
            답변에는 한자가 나오지 않도록 해주십시오.

            [지시 사항]:
            1. 너의 주관적인 의견은 배제하고, 반드시 제공된 [참고 문헌]의 내용을 기반으로 답변해.
            2. 답변 과정에서 어떤 파일(txt, pdf)을 참고했는지 명확히 언급해.
            3. [참고 문헌]에 질문과 관련된 내용이 없으면 "제공된 문서 내에서 관련 정보를 찾을 수 없습니다"라고 정중히 답해.

            [참고 문헌]:
            {context_text}

            [질문]: {query}

            답변:
            """
            
            completion = st.session_state.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.1 # 답변의 일관성을 위해 낮게 설정
            )
            response = completion.choices[0].message.content
            
            # 최종 답변 및 하단 출처 표시
            source_links = f"\n\n---\n**📍 참고된 파일들:** {', '.join(sources)}"
            full_response = f"{response}{source_links}"
            
            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})