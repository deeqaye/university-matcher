from django.test import TestCase
from django.urls import reverse

class UniversityMatchTests(TestCase):

    def test_index_view(self):
        response = self.client.get(reverse('universities:index'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'universities/index.html')

    def test_results_view(self):
        response = self.client.get(reverse('universities:results'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'universities/results.html')

    def test_match_score_calculation(self):
        # Assuming you have a function to calculate match scores
        # This is a placeholder for actual match score logic
        user_input = {
            'gpa': 3.5,
            'sat': 1200,
            'budget_max': 30000,
            'countries': ['UK'],
            'languages': ['English'],
        }
        # Call your match score function here and assert the expected output
        # Example: score = calculate_match_score(user_input)
        # self.assertEqual(score, expected_score)