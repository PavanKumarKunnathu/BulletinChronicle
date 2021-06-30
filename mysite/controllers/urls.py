from django.urls import path
from . import views

from django.conf import settings
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf.urls.static import static


urlpatterns=[
    path('',views.createnews,name="createnews"),
    path('addnews/',views.addnews,name="addnews"),
    path('getsummary',views.getsummary,name="getsummary"),

    ]

urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)