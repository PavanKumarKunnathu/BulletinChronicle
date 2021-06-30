from django.urls import path
from . import views

from django.conf import settings
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.conf.urls.static import static
# path('sportsNews',views.sportsNews,name="sportsNews"),/
urlpatterns=[
    path('',views.index,name="index"),
    path('getsummary',views.getsummary,name="getsummary"),
    path('langconverter',views.langconverter,name="langconverter"),
    path('faceverification/',views.faceverification,name="faceverification"),
    path('getfacename/',views.getfacename,name="getfacename"),
    path('analyseuser/',views.analyse_user,name="analyse_user"),


    path('sportsNews',views.sportsNews,name="sportsNews"),
    path('sports/<str:sport_id>/',views.sports,name="sports"),

    path('technologyNews', views.technologyNews, name="technologyNews"),
    path('technology/<str:technology_id>/', views.technology, name="technology"),

    path('businessNews', views.businessNews, name="businessNews"),
    path('business/<str:business_id>/', views.business, name="business"),

    path('educationNews', views.educationNews, name="educationNews"),
    path('education/<str:education_id>/', views.education, name="education"),

    path('entertainmentNews', views.entertainmentNews, name="entertainmentNews"),
    path('entertainment/<str:entertainment_id>/', views.entertainment, name="entertainment"),

    path('socialProblems',views.socialproblemsNews,name="socialproblemsNews"),
    path('socialproblem/<str:sp_id>/',views.socialproblems,name="socialproblems"),

    path('login',views.login,name="login"),
    path('loginauth/',views.loginauth,name="loginauth"),
    path('faceauthentication',views.faceauthentication,name="faceauthentication"),

    path('confirmfaceverification/',views.confirmfaceverification,name="confirmfaceverification"),
    path('confirmprofile/',views.confirmProfile,name="confirmprofile"),
    path('analyseUserPosts/',views.analyse_user_posts,name="analyse_user_posts"),
    path('checksessionid/',views.checksessionid,name="checksessionid"),

    path('navbar/',views.navbar,name="navbar"),
    path('addpost/',views.addpost,name="addpost"),
    path('user_notifications/',views.user_notifications,name="user_notifications"),
    path('toxicCommentClassification/',views.toxiccomment,name="toxiccomment"),
    path('logout/',views.logout,name="logout"),
    path('checktoxiccomment',views.checktoxiccomment,name="checktoxiccomment"),
    path('clicklike/',views.clicklike,name="clicklike"),
    path('deletecomment/',views.deletecomment,name="deletecomment"),

    path('supportpost/',views.supportpost,name="supportpost"),



    path('profileVerification/',views.profileVerification,name="profileVerification"),
    path("ProfileVerificationUnsucess",views.ProfileVerificationUnsucess,name="ProfileVerificationUnsucess")
    ]

urlpatterns += staticfiles_urlpatterns()
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)