import os
import json
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Typesense
import typesense
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from src.utils.agent import render_template
from .models import AnswerQuestionResponse, ExplanationState, Question, OptionExplanation, ReviewAnswerResponse, ReformulateContextResponse, TrainingReference, to_kebab_case, ClassifyReferencesResponse, ReferenceClassification

load_dotenv()


class Workflow:
    def __init__(self):

        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if anthropic_api_key:
            from langchain_anthropic import ChatAnthropic
            self.llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0.2)
        elif openai_api_key:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        else:
            raise ValueError("No supported LLM API key found. Please set ANTHROPIC_API_KEY or OPENAI_API_KEY.")

        embedding = OpenAIEmbeddings(model="text-embedding-3-small")

        self.embedding_model = embedding

        node = {
          "host": os.getenv("TYPESENSE_HOST", "localhost"),
          "port": os.getenv("TYPESENSE_PORT", "8108"),
          "protocol": os.getenv("TYPESENSE_PROTOCOL", "http")
        }
        typesense_client = typesense.Client(
            {
              "nodes": [node],
              "api_key": os.getenv("TYPESENSE_API_KEY"),
              "connection_timeout_seconds": 2
            }
        )
        typesense_collection_name = "exercises"

        self.vectorstore = Typesense(
            typesense_client=typesense_client,
            embedding=embedding,
            typesense_collection_name=typesense_collection_name,
            text_key="content"
        )

        self.workflow = self._build_workflow()

    def _build_workflow(self):
        graph = StateGraph(ExplanationState)

        # ==================== Nodes Setup ====================

        graph.add_node("start", lambda state: state)
        graph.add_node("get_training_context", self._get_training_context)
        graph.add_node("classify_references", self._classify_references)
        graph.add_node("reformulate_context", self._reformulate_context)
        graph.add_node("answer_option", self._answer_option)
        graph.add_node("review_answer", self._review_answer)
        graph.add_node("advance_option", self._advance_option)
        graph.add_node("continue_reviewing", self._continue_reviewing)
        graph.add_node("finalize", self._finalize)

        # ==================== Edges Setup ====================

        graph.set_entry_point("start")

        graph.add_edge("start", "get_training_context")
        graph.add_edge("get_training_context", "classify_references")
        graph.add_edge("classify_references", "reformulate_context")
        graph.add_edge("reformulate_context", "answer_option")
        graph.add_edge("answer_option", "review_answer")
        
        graph.add_conditional_edges(
            "review_answer",
            self._should_continue_reviewing,
            {
                "continue_reviewing": "continue_reviewing",
                "advance_option": "advance_option",
                "complete": "finalize"
            }
        )

        graph.add_edge("continue_reviewing", "answer_option")
        graph.add_edge("advance_option", "answer_option")
        graph.add_edge("finalize", END)

        compiled_graph = graph.compile()

        try:
          compiled_graph.get_graph().draw_mermaid_png(output_file_path="workflow.png")
        except Exception as e:
          print(f"Could not save workflow graph image: {e}")

       
        return compiled_graph

    async def _get_training_context(self, state: ExplanationState) -> ExplanationState:
        # Build filter string
        filter_parts = ["metadata.type: LECTURE"]
        
        if state.certification_id:
            filter_parts.append(f"metadata.certification_id: {state.certification_id}")
        
        if state.tech:
            filter_parts.append(f"metadata.tech: {state.tech}")
            
        if state.tech_id:
            filter_parts.append(f"metadata.tech_id: {state.tech_id}")
            
        if state.training_slug:
            filter_parts.append(f"metadata.training_slug: {state.training_slug}")
        
        filter_string = " && ".join(filter_parts)
        
        docs = self.vectorstore.similarity_search_with_score(
            f"{state.question.title}\n{state.question.description}",
            k=5,
            filter=filter_string
        )

        training_references = []
        for doc, score in docs:
            metadata = doc.metadata
            chapter = metadata.get('chapter', 'Unknown')
            title = metadata.get('title', 'Untitled')
            training_slug = metadata.get('training_slug', '')
            tech = metadata.get('tech', '')
            sort_order = metadata.get('sort_order', '')
            
            if chapter == "Unknown" or not training_slug or not tech:
                continue
                
            url = f"/{tech}/dashboard/training/{training_slug}?chapter={to_kebab_case(chapter)}"
            if sort_order:
                url += f"&part={sort_order}"
            
            training_ref = TrainingReference(
                title=title,
                chapter=chapter,
                training_slug=training_slug,
                tech=tech,
                url=url,
                content=doc.page_content,
                similarity_score=score
            )
            training_references.append(training_ref)

        # Remove duplicates based on chapter and title combination
        seen_combinations = set()
        deduplicated_references = []
        for ref in training_references:
            combination = (ref.chapter, ref.title)
            if combination not in seen_combinations:
                seen_combinations.add(combination)
                deduplicated_references.append(ref)
        
        state.training_references = deduplicated_references
        return state

    async def _classify_references(self, state: ExplanationState) -> ExplanationState:
        
        if not state.training_references:
            return state
        
        formatted_options = ""
        for i, option in enumerate(state.question.options):
            letter = chr(ord('A') + i)
            formatted_options += f"{letter}. {option.option}\n"
        
        correct_answer = None
        for i, option in enumerate(state.question.options):
            if option.is_correct:
                correct_answer = chr(ord('A') + i) + ". " + option.option
                break
        
        references = []
        for ref in state.training_references:
            references.append({
                "title": ref.title,
                "chapter": ref.chapter,
                "similarity_score": ref.similarity_score,
                "content": ref.content
            })
        
        template_vars = {
            "question": f"{state.question.title}\n{state.question.description}",
            "formatted_options": formatted_options,
            "correct_answer": correct_answer,
            "references": references
        }
        
        prompt = render_template("classify_references.j2", template_vars)
        
        structured_llm = self.llm.with_structured_output(ClassifyReferencesResponse)
        resp = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        
        relevant_indices = []
        for classification in resp.classifications:
            if classification.classification == "RELEVANT":
                relevant_indices.append(classification.reference_number - 1)
        
        filtered_references = []
        for i in relevant_indices:
            if i < len(state.training_references):
                filtered_references.append(state.training_references[i])
        
        if filtered_references:
            state.training_references = filtered_references
        
        if filtered_references:
            context = ""
            for ref in filtered_references:
                context += "=" * 40 + "\n"
                context += f"Title: {ref.title}\n"
                context += f"Chapter: {ref.chapter}\n"
                context += f"{ref.content}\n"
                context += "=" * 40 + "\n\n"
            
            state.context = context
            
            docs_by_chapter = {}
            for ref in filtered_references:
                if ref.chapter not in docs_by_chapter:
                    docs_by_chapter[ref.chapter] = []
                
                from langchain_core.documents import Document
                doc = Document(
                    page_content=ref.content,
                    metadata={
                        'title': ref.title,
                        'chapter': ref.chapter,
                        'training_slug': ref.training_slug,
                        'tech': ref.tech,
                        'type': 'LECTURE'
                    }
                )
                docs_by_chapter[ref.chapter].append(doc)
            
            state.docs_by_chapter = docs_by_chapter
        else:
            # If no filtered references, set empty defaults
            state.context = ""
            state.docs_by_chapter = {}
        
        return state

    async def _reformulate_context(self, state: ExplanationState) -> ExplanationState:
        
        formatted_options = ""
        for i, option in enumerate(state.question.options):
            letter = chr(ord('A') + i)
            formatted_options += f"{letter}. {option.option}\n"
        
        correct_answer = None
        for i, option in enumerate(state.question.options):
            if option.is_correct:
                correct_answer = chr(ord('A') + i) + ". " + option.option
                break
        
        template_vars = {
            "question": f"{state.question.title}\n{state.question.description}",
            "formatted_options": formatted_options,
            "correct_answer": correct_answer,
            "docs_by_chapter": state.docs_by_chapter
        }
        
        prompt = render_template("reformulate_context.j2", template_vars)

        structured_llm = self.llm.with_structured_output(ReformulateContextResponse)
        resp = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        
        state.context = resp.reformulated_context
        
        return state

    async def _answer_option(self, state: ExplanationState) -> ExplanationState:
        
        current_option = state.question.options[state.current_option_index]
        
        formatted_options = ""
        for i, option in enumerate(state.question.options):
            letter = chr(ord('A') + i)
            formatted_options += f"{letter}. {option.option}\n"
        
        correct_answer = None
        for i, option in enumerate(state.question.options):
            if option.is_correct:
                correct_answer = chr(ord('A') + i) + ". " + option.option
                break
        
        student_choice = chr(ord('A') + state.current_option_index) + ". " + current_option.option
        
        template_vars = {
            "question": f"{state.question.title}\n{state.question.description}",
            "formatted_options": formatted_options,
            "context": state.context,
            "is_correct": current_option.is_correct,
            "correct_answer": correct_answer,
            "incorrect_answer": student_choice
        }
        
        if state.current_review:
            template_vars["previous_answer"] = state.current_answer
            template_vars["review_feedback"] = state.current_review
        
        prompt = render_template("answer_question.j2", template_vars)
        
        structured_llm = self.llm.with_structured_output(AnswerQuestionResponse)
        resp = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        
        state.current_answer = resp.explanation
        return state

    async def _review_answer(self, state: ExplanationState) -> ExplanationState:
        
        
        current_option = state.question.options[state.current_option_index]
        
        
        formatted_options = ""
        for i, option in enumerate(state.question.options):
            letter = chr(ord('A') + i)
            formatted_options += f"{letter}. {option.option}\n"
        
        
        correct_answer = None
        for i, option in enumerate(state.question.options):
            if option.is_correct:
                correct_answer = chr(ord('A') + i)
                break
        
        
        student_choice = chr(ord('A') + state.current_option_index)
        
        
        template_vars = {
            "question": f"{state.question.title}\n{state.question.description}",
            "options": formatted_options,
            "correct_answer": correct_answer,
            "answer": student_choice,
            "is_correct": current_option.is_correct,
            "explanation": state.current_answer,
            "formatted_relevant_docs": state.context  
        }
        
        prompt = render_template("review_answer.j2", template_vars)

        structured_llm = self.llm.with_structured_output(ReviewAnswerResponse)
        
        resp = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        
        try:

            is_approved = resp.is_approved
            review_text = resp.review
            
            if is_approved or state.max_retries == 0:
                
                explanation = OptionExplanation(
                    option=current_option.option,
                    is_correct=current_option.is_correct,
                    explanation=state.current_answer,
                )
                state.option_explanations.append(explanation)
                state.current_review = None
                state.current_answer = None
            else:
                
                state.current_review = review_text
                
        except json.JSONDecodeError:
            
            state.current_review = "Review format error. Please regenerate the explanation."
        
        return state

    def _continue_reviewing(self, state: ExplanationState) -> ExplanationState:
        if state.current_review and state.max_retries > 0:
            state.max_retries = state.max_retries - 1
        
        return state

    def _should_continue_reviewing(self, state: ExplanationState) -> str:
        if state.current_review and state.max_retries > 0:
            return "continue_reviewing"
        
        if state.current_option_index < len(state.question.options) - 1:
            return "advance_option"
        
        return "complete"

    async def _advance_option(self, state: ExplanationState) -> ExplanationState:
        state.max_retries = 3
        state.current_answer = None
        state.current_review = None
        state.current_option_index = state.current_option_index + 1
        return state

    async def _finalize(self, state: ExplanationState) -> ExplanationState:
        state.is_complete = True
        return state



    async def run(self, question: Question, certification_id: Optional[str] = None, 
                  tech: Optional[str] = None, tech_id: Optional[str] = None, 
                  training_slug: Optional[str] = None) -> ExplanationState:
        initial_state = ExplanationState(
            question=question,
            certification_id=certification_id,
            tech=tech,
            tech_id=tech_id,
            training_slug=training_slug
        )
        final_state = await self.workflow.ainvoke(initial_state, {
            "recursion_limit": 120
        })
        return ExplanationState(**final_state)
