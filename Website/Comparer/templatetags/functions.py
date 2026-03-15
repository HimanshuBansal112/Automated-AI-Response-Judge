from django import template

register = template.Library()

@register.filter
def index(lst, i):
    try:
        return lst[i]
    except:
        return ""

@register.filter
def to_percent(value):
    try:
        return f"{float(value) * 100:.2f}"
    except (ValueError, TypeError):
        return value