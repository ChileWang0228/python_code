from django.template import loader
from .models import Question
from django.shortcuts import render, get_object_or_404
# Create your views here.


def index(request):
    # return HttpResponse('Hello, Chile. You are at the polls index')
    latest_question_list = Question.objects.order_by('-pub_date')[:5]
    context = {
        'latest_question_list': latest_question_list,
    }
    # return HttpResponse(template.render(context, request))
    """
    render() 函数把请求(HttpRequest)对象作为第一个参数，加载的模版名字作为第二个参数，
    用于渲染模板的上下文字典作为可选的第三个参数。函数返回一个 HttpResponse 对象，
    内容为指定模板用指定上下文渲染后的结果。
    """
    return render(request, 'polls/index.html', context)


def detail(request, question_id):
    question = get_object_or_404(Question, pk=question_id)  # 若行则执行，不行404
    return render(request, 'polls/detail.html', {'question': question})

