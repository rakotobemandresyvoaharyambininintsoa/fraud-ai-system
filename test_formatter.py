from utils.shap_formatter import format_shap_features


data = [

{
"feature":"remainder__amount",
"impact":0.2584
},

{
"feature":"categorical__merchant_unknown",
"impact":0.1675
}

]


result = format_shap_features(data)

print(result)