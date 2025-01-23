from django.urls import path
from reader import views

urlpatterns = [
    path('', views.home, name="Home"),
    path('start', views.start, name="Start"),
    path('reuploadfile',views.reuploadfile,name="reuploadfile"), 
    path('savefile',views.savefile,name="savefile"),     
    path('explore-count', views.explorecount, name="explorecount"), 
    path('explore-confusion-matrix/<str:atype>',views.exploreconfusionmatrix,name="exploreconfusionmatrix"),   
    path('profilecheck',views.profilecheck,name="profilecheck"), 
    path('uploadprofilecheck',views.uploadprofilecheck,name="uploadprofilecheck"), 
]