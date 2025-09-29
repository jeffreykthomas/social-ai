"""frontend_server URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.conf.urls import include, url
from django.urls import path, re_path
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static

from translator import views as translator_views
from views_agent_monitor import (
    agent_monitor_view,
    get_agent_states,
    get_agent_details,
    update_agent_need,
    trigger_agent_event,
    get_prediction_explanation,
    get_learning_progress,
)

urlpatterns = [
    url(r'^$', translator_views.landing, name='landing'),
    url(r'^simulator_home$', translator_views.home, name='home'),
    url(r'^demo/(?P<sim_code>[\w-]+)/(?P<step>[\w-]+)/(?P<play_speed>[\w-]+)/$', translator_views.demo, name='demo'),
    url(r'^replay/(?P<sim_code>[\w-]+)/(?P<step>[\w-]+)/$', translator_views.replay, name='replay'),
    url(r'^replay_persona_state/(?P<sim_code>[\w-]+)/(?P<step>[\w-]+)/(?P<persona_name>[\w-]+)/$', translator_views.replay_persona_state, name='replay_persona_state'),
    url(r'^process_environment/$', translator_views.process_environment, name='process_environment'),
    url(r'^update_environment/$', translator_views.update_environment, name='update_environment'),
    url(r'^path_tester/$', translator_views.path_tester, name='path_tester'),
    url(r'^path_tester_update/$', translator_views.path_tester_update, name='path_tester_update'),
    path('admin/', admin.site.urls),

    # Predictive agents monitor and APIs
    path('agent_monitor/', agent_monitor_view, name='agent_monitor'),
    path('api/agent_states/', get_agent_states, name='api_agent_states'),
    path('api/agent/<str:agent_name>/', get_agent_details, name='api_agent_details'),
    path('api/agent/<str:agent_name>/need/', update_agent_need, name='api_update_agent_need'),
    path('api/trigger_event/', trigger_agent_event, name='api_trigger_event'),
    path('api/agent/<str:agent_name>/predictions/', get_prediction_explanation, name='api_agent_predictions'),
    path('api/agent/<str:agent_name>/learning/', get_learning_progress, name='api_agent_learning'),
]
