from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages

# Provides a simple user register form 
def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("/user/login")
        else:
            return render(request, "user/register.html", {"form": form})

    else:
        form = UserCreationForm()
        return render(request, "user/register.html", {"form": form})