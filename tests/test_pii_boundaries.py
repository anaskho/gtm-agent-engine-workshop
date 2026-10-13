import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["OPENAI_API_KEY"] = "test-key"

from gtm_agent import data_service, gtm_agent


class ProspectPIIBoundaryTests(unittest.TestCase):
    def setUp(self):
        data_service._PROFILES.clear()

    def assert_no_billing_data(self, output):
        serialized = json.dumps(output)
        self.assertNotIn("billing_qualification", serialized)
        self.assertNotIn("541-88-2307", serialized)
        self.assertNotIn("1984-03-11", serialized)
        self.assertNotIn("4539882100437712", serialized)

    def test_get_prospect_returns_only_contact_allowlist(self):
        record = {
            "name": "Omar Okafor",
            "email": "omar@example.com",
            "disqualified": False,
            "engagement_history": [{"campaign": "demo"}],
            "account_details": [{"account": "Example"}],
            "tech_stack": ["Kafka"],
            "annual_revenue": 90000000,
            "billing_qualification": {
                "tax_id": "541-88-2307",
                "date_of_birth": "1984-03-11",
                "card_on_file": "4539882100437712",
            },
        }
        with patch.object(data_service, "get_prospect_record", return_value=record):
            output = gtm_agent.get_prospect.invoke({"prospect_id": "LEAD-1"})

        self.assertEqual(
            output,
            {
                "prospect": {
                    "prospect_id": "LEAD-1",
                    "name": "Omar Okafor",
                    "email": "omar@example.com",
                    "disqualified": False,
                },
                "found": True,
            },
        )
        self.assert_no_billing_data(output)

    def test_fresh_profile_is_filtered_before_caching_and_returning(self):
        record = {
            "name": "Omar Okafor",
            "email": "omar@example.com",
            "phone": "555-0100",
            "disqualified": False,
            "annual_revenue": 90000000,
            "enrichment_source": "Example",
            "billing_qualification": {
                "tax_id": "541-88-2307",
                "date_of_birth": "1984-03-11",
                "card_on_file": "4539882100437712",
            },
        }
        with (
            patch.object(data_service, "get_prospect_record", return_value=record),
            patch.object(data_service, "fetch_engagement_history", return_value=[{"campaign": "demo"}]),
            patch.object(data_service, "fetch_account_details", return_value=[{"account": "Example"}]),
            patch.object(data_service, "fetch_tech_stack", return_value=["Kafka"]),
        ):
            output = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

        profile = output["prospect_profile"]
        self.assertEqual(set(profile), set(data_service.PROSPECT_PROFILE_FIELDS))
        self.assertEqual(data_service._PROFILES["LEAD-1"], profile)
        self.assert_no_billing_data(output)

    def test_cached_profile_is_filtered_before_returning(self):
        data_service._PROFILES["LEAD-1"] = {
            "prospect_id": "LEAD-1",
            "name": "Omar Okafor",
            "account_details": [{"account": "Example"}],
            "annual_revenue": 90000000,
            "tech_stack": ["Kafka"],
            "engagement_history": [{"campaign": "demo"}],
            "email": "omar@example.com",
            "tax_id": "541-88-2307",
            "billing_qualification": {
                "tax_id": "541-88-2307",
                "date_of_birth": "1984-03-11",
                "card_on_file": "4539882100437712",
            },
        }

        output = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

        profile = output["prospect_profile"]
        self.assertEqual(set(profile), set(data_service.PROSPECT_PROFILE_FIELDS))
        self.assertEqual(data_service._PROFILES["LEAD-1"], profile)
        self.assert_no_billing_data(output)

    def test_score_prompt_contains_only_scoring_fields(self):
        profile = {
            "prospect_id": "LEAD-1",
            "name": "Omar Okafor",
            "account_details": [{"account": "Example"}],
            "annual_revenue": 90000000,
            "tech_stack": ["stale"],
            "engagement_history": [{"campaign": "demo"}],
            "email": "omar@example.com",
            "tax_id": "541-88-2307",
            "billing_qualification": {
                "tax_id": "541-88-2307",
                "date_of_birth": "1984-03-11",
                "card_on_file": "4539882100437712",
            },
        }
        offering = {
            "required_tech_stack": ["Kafka"],
            "min_annual_revenue": 1000000,
            "description": "Example offering",
        }
        result = SimpleNamespace(model_dump=lambda: {"score": 80})
        scoring_llm = SimpleNamespace(invoke=Mock(return_value=result))
        with (
            patch.object(data_service, "fetch_tech_stack", return_value=["Kafka"]),
            patch.object(gtm_agent, "_scoring_llm", scoring_llm),
        ):
            output = gtm_agent.score_prospect.invoke(
                {"prospect_profile": profile, "offering": offering}
            )

        user_prompt = scoring_llm.invoke.call_args.args[0][1]["content"]
        serialized_profile = user_prompt.split("\n\nProspect profile:\n", 1)[1]
        sent_profile = json.loads(serialized_profile)
        self.assertEqual(
            set(sent_profile),
            {
                "prospect_id",
                "name",
                "account_details",
                "annual_revenue",
                "tech_stack",
                "engagement_history",
            },
        )
        self.assertEqual(sent_profile["tech_stack"], ["Kafka"])
        self.assert_no_billing_data(sent_profile)
        self.assert_no_billing_data(output)


if __name__ == "__main__":
    unittest.main()
