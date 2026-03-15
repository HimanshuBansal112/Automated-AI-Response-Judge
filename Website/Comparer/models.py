from django.db import models

class PromptRecord(models.Model):
    prompt = models.JSONField()
    response_a = models.JSONField()
    response_b = models.JSONField()

class EvaluationResult(models.Model):
    chat_input = models.OneToOneField(PromptRecord, related_name='result', on_delete=models.CASCADE)
    fact_score_a = models.FloatField()
    fact_score_b = models.FloatField()
    fact_tie = models.FloatField()
    style_score_a = models.FloatField()
    style_score_b = models.FloatField()
    style_tie = models.FloatField()