$(function(){
$('button').click(function(){
    var global_model = $('#global_model').val();
    var text_review = $('#input_value').val();
    $.ajax({
        url: '/squad_predict',
        data: $('form').serialize(),
        type: 'POST',
        success: function(response){
            $('#msg').removeClass('hide');
            $('#msg').html('<div class="alert alert-success alert-outline alert-dismissible" role="alert"> Answer: '+ response + '</div>');
            $('#msg').addClass('success');
        },
        error: function(error){
            $('#msg').removeClass('hide');
            $('#msg').html('<div class="alert alert-danger alert-outline alert-dismissible" role="alert">'+ error +'</div>');
            $('#msg').addClass('error');
        }
    });
});
});