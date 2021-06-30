from django.contrib import admin

# Register your models here.
from .models import users,WebcamPictures,temp,demoposts,posts,notifications,newscomments,likes,supportnews

# Register your models here.
admin.site.register(users)
admin.site.register(WebcamPictures)
admin.site.register(temp)
admin.site.register(demoposts)
admin.site.register(posts)
admin.site.register(notifications)
admin.site.register(newscomments)
admin.site.register(likes)
admin.site.register(supportnews)

