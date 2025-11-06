from django.test import TestCase
from django.urls import reverse

class GeminiViewTests(TestCase):
    def test_inquire_view(self):
        response = self.client.get(reverse('gemini:inquire'))
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'gemini/inquire.html')

    def test_inquire_form_submission(self):
        response = self.client.post(reverse('gemini:inquire'), {
            'model': 'Gemini 2.5-Pro',
            'query': 'What are the features?',
        })
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Features of Gemini 2.5-Pro')  # Adjust based on actual response content

    def test_image_extraction(self):
        # This is a placeholder for testing image extraction functionality
        # You would implement the actual logic to extract images from the website
        self.assertTrue(True)  # Replace with actual test logic for image extraction