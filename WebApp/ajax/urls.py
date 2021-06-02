from django.urls import path
from . import views

urlpatterns = [
    path('get_columns/<int:id>', views.get_columns, name='get_columns'),
    path('get_agent_type/<int:id>', views.get_agent_type, name='get_agent_type'),
    path('get_dedupe_pair/', views.get_dedupe_pair, name='get_dedupe_pair'),
    path('store_dedupe_training/', views.store_dedupe_training, name='store_dedupe_training'),
    path('get_other_gensim_agents/<int:id>', views.get_other_gensim_agents, name='get_other_gensim_agents')
]
