import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent import gtm_agent


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "REP-001"}})
        self.prospect = {"prospect_id": "LEAD-50001", "email": "lead@example.com", "name": "Lead"}

    @patch("gtm_agent.gtm_agent.uuid.uuid4")
    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record", return_value={"disqualified": True})
    def test_disqualified_prospect_is_blocked_before_message_id(self, get_record, uuid4):
        result = gtm_agent.send_prospect_email.func(self.prospect, "Subject", "Body", self.runtime)

        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect LEAD-50001 is disqualified; email not sent.",
        })
        get_record.assert_called_once_with("LEAD-50001")
        uuid4.assert_not_called()

    @patch("gtm_agent.gtm_agent.uuid.uuid4")
    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record", return_value=None)
    def test_missing_prospect_is_blocked_before_message_id(self, get_record, uuid4):
        result = gtm_agent.send_prospect_email.func(self.prospect, "Subject", "Body", self.runtime)

        self.assertEqual(result, {
            "status": "blocked",
            "error": "Prospect LEAD-50001 was not found in the source record; email not sent.",
        })
        get_record.assert_called_once_with("LEAD-50001")
        uuid4.assert_not_called()

    @patch("gtm_agent.gtm_agent.uuid.uuid4")
    @patch("gtm_agent.gtm_agent.data_service.get_prospect_record", return_value={"disqualified": False})
    def test_eligible_prospect_is_sent(self, get_record, uuid4):
        uuid4.return_value.hex = "0123456789abcdef"

        result = gtm_agent.send_prospect_email.func(self.prospect, "Subject", "Body", self.runtime)

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["message_id"], "msg-0123456789ab")
        self.assertEqual(result["to"], "lead@example.com")
        get_record.assert_called_once_with("LEAD-50001")
        uuid4.assert_called_once_with()

    def test_system_prompt_explains_blocked_email(self):
        self.assertIn("first call get_prospect before drafting or sending", gtm_agent.SYSTEM_PROMPT)
        self.assertIn("clearly explain to the rep why the email was not sent", gtm_agent.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
