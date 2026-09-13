"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant, trợ lý AI của hệ sinh thái Vingroup.

## PERSONA
- Vai trò: tư vấn sản phẩm VinFast, dịch vụ Vinpearl và tiếp nhận hỗ trợ.
- Phong cách: chuyên nghiệp, thân thiện, ngắn gọn và chính xác.

## AVAILABLE TOOLS
- search_product_catalog: tra cứu danh mục xe điện hoặc du lịch theo mức giá.
- submit_support_ticket: tạo phiếu hỗ trợ và trả về mã ticket chính thức.

## CORE RULES
1. Không bịa tên sản phẩm, giá, tình trạng hoặc mã ticket.
2. Phải gọi search_product_catalog khi người dùng yêu cầu tìm sản phẩm theo danh mục/giá.
3. Phải gọi submit_support_ticket khi người dùng muốn báo lỗi hoặc ghi nhận phản hồi.
4. Một yêu cầu có thể cần cả hai tool; xử lý đầy đủ từng intent.
5. Nếu tool không trả về kết quả, thông báo rõ rằng không tìm thấy lựa chọn phù hợp.

## OPERATIONAL BOUNDARIES
- Chỉ hỗ trợ nội dung liên quan đến sản phẩm và dịch vụ trong hệ sinh thái Vingroup.
- Với thông tin không có trong tool hoặc dữ liệu được cung cấp, nói rõ giới hạn và không suy đoán.

## OUTPUT CONTRACT
- Nội bộ tuân theo chu trình Thought → Action → Observation → Final Answer.
- Không tiết lộ suy luận nội bộ. Câu trả lời cuối chỉ trình bày kết quả, dữ liệu tool và bước tiếp theo hữu ích.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    @staticmethod
    def _detect_intents(user_input: str) -> Dict[str, bool]:
        """Nhận diện độc lập các intent để một câu có thể dùng nhiều tool."""
        text = user_input.lower()
        catalog_keywords = (
            "giá dưới", "dưới", "bao nhiêu tiền", "tìm xe",
            "resort", "khách sạn", "du lịch", "xem xe",
        )
        ticket_keywords = (
            "bị lỗi", "lỗi hệ thống", "hỗ trợ", "xử lý gấp",
            "nghiêm trọng", "ghi nhận phản hồi", "phản hồi", "ẩm mốc",
        )
        return {
            "needs_catalog": any(keyword in text for keyword in catalog_keywords),
            "needs_ticket": any(keyword in text for keyword in ticket_keywords),
        }

    @staticmethod
    def _catalog_arguments(user_input: str) -> Dict[str, Any]:
        text = user_input.lower()
        category = "du_lich" if any(
            word in text for word in ("vinpearl", "resort", "khách sạn", "du lịch", "phòng")
        ) else "xe_dien"

        price_match = re.search(r"(\d+(?:[.,]\d+)?)\s*(triệu|tỷ)", text)
        max_price = 999_999_999_999
        if price_match:
            amount = float(price_match.group(1).replace(",", "."))
            multiplier = 1_000_000_000 if price_match.group(2) == "tỷ" else 1_000_000
            max_price = int(amount * multiplier)

        return {"category": category, "max_price": max_price}

    @staticmethod
    def _ticket_arguments(user_input: str) -> Dict[str, str]:
        name_match = re.search(
            r"(?:tôi tên|tên tôi là)\s+([^,.!?]+)",
            user_input,
            flags=re.IGNORECASE,
        )
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

        text = user_input.lower()
        if any(word in text for word in ("nghiêm trọng", "gấp", "khẩn cấp")):
            priority = "high"
        elif any(word in text for word in ("mức độ thấp", "ưu tiên thấp")):
            priority = "low"
        else:
            priority = "medium"

        issue_description = user_input.strip()
        return {
            "customer_name": customer_name,
            "issue_description": issue_description,
            "priority": priority,
        }

    @staticmethod
    def _format_products(products: List[Dict[str, Any]]) -> str:
        if not products:
            return "Rất tiếc, không tìm thấy sản phẩm phù hợp."
        if products and "error" in products[0]:
            return f"Không thể tra cứu danh mục: {products[0]['error']}"

        lines = ["Các lựa chọn phù hợp:"]
        for product in products:
            price = f"{product['price_vnd']:,}".replace(",", ".")
            lines.append(f"- {product['name']}: {price} VNĐ")
        return "\n".join(lines)

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intents(user_input)
        actions = []
        if intents["needs_catalog"]:
            actions.append(("search_product_catalog", self._catalog_arguments(user_input)))
        if intents["needs_ticket"]:
            actions.append(("submit_support_ticket", self._ticket_arguments(user_input)))

        # FAQ là một vòng xử lý trực tiếp, không gọi tool.
        if not actions:
            if self.max_iterations < 1:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": 0,
                    "status": "max_iterations_reached",
                }
            answer = (
                "Chính sách bảo hành pin xe điện VinFast phụ thuộc vào mẫu xe và "
                "điều kiện áp dụng. Vui lòng kiểm tra chính sách bảo hành hiện hành "
                "hoặc liên hệ VinFast để nhận thông tin chính xác."
            )
            self.trace.append({
                "iteration": 1,
                "action": "answer_faq",
                "observation": answer,
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed",
            }

        observations = []
        iteration = 0
        for tool_name, arguments in actions:
            if iteration >= self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "max_iterations_reached",
                }

            iteration += 1
            result = TOOL_MAP[tool_name](**arguments)
            self.trace.append({
                "iteration": iteration,
                "action": tool_name,
                "arguments": arguments,
                "observation": result,
            })

            if tool_name == "search_product_catalog":
                observations.append(self._format_products(result))
            else:
                observations.append(
                    f"Đã tạo ticket {result['ticket_id']} cho "
                    f"{result['customer_name']} với mức ưu tiên {result['priority']}."
                )

        return {
            "answer": "\n\n".join(observations),
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed",
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
