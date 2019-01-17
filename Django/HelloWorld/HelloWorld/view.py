from django.shortcuts import render


def hello(request):
	context = {}
	context['hello'] = 'Hello Chile!'
	return render(request, 'hello.html', context)
