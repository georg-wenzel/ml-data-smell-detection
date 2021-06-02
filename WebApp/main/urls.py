from django.urls import path
from . import views, views_agents, views_datasets

urlpatterns = [
    path('datasets/delete', views_datasets.delete_dataset, name='delete_dataset'),
    path('datasets/add', views_datasets.add_dataset, name='add_dataset'),
    path('datasets/<int:id>', views_datasets.edit_dataset, name='edit_dataset'),
    path('datasets/', views_datasets.datasets, name='datasets'),
    path('agents/add', views_agents.add_agent, name='add_agent'),
    path('agents/train/<int:id>', views_agents.train_agent, name='train_agent'),
    path('agents/delete/', views_agents.delete_agent, name='delete_agent'),
    path('agents/<int:id>', views_agents.edit_agent, name='edit_agent'),
    path('agents/', views_agents.agents, name='agents'),
    path('analyze/duplicates', views.duplicates, name='duplicates'),
    path('analyze/', views.analyze, name='analyze'),
    path('download/', views.download, name='download'),
    path('download_model/<int:id>', views.download_model, name='download_model'),
    path('download_settings/<int:id>', views.download_settings, name='download_settings'),
    path('', views.home, name='home')
]
