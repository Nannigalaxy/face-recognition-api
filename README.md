# face embedding API

API returns face embedding vector as response.

## Getting Started

This API uses deep learning to generate face embedding vector using Keras on top of tensorflow.\
Facenet implementation. Inception-ResNet-v2 model.

### Prerequisites

```
pip install -r requirements.txt
```

### Run the server

```
gunicorn api:app
```

## Example 

### Request

/face-embedding/url

```
curl -i -H 'Content-Type: application/json' \
-d '{"url":"https://i.imgur.com/oRa0KpU.jpeg"}' \
http://localhost:8000/face-embedding/url
```
### Response

```
HTTP/1.1 201 CREATED
Server: gunicorn/20.0.4
Date: Mon, 20 Apr 2020 18:55:52 GMT
Connection: close
Content-Type: application/json
Content-Length: 2581

{"face_embedding":{"url":"https://i.imgur.com/oRa0KpU.jpeg","vector":[1.0644716024398804,-0.7762305736541748,-1.5508257150650024,0.09768518060445786,3.234025001525879,0.7380242943763733,0.7755756378173828,0.7783681154251099,1.7994431257247925,1.3094197511672974,0.11752720177173615,-0.9972988367080688,-1.387189269065857,-1.0655053853988647,0.6015282869338989,-0.6466184854507446,-0.41857847571372986,-0.10470283031463623,0.4276144504547119,0.7561772465705872,1.6428494453430176,0.7238189578056335,-0.4318429231643677,0.4909619688987732,0.6246815919876099,-0.16436511278152466,0.643584668636322,-1.6872013807296753,-0.41700923442840576,-0.7216029763221741,1.135551929473877,-0.08613882958889008,-0.5983056426048279,-0.36481523513793945,-0.5517364740371704,0.3539840877056122,0.009662304073572159,-0.9530348181724548,-0.3983016014099121,-1.7624664306640625,-1.8313381671905518,0.34050893783569336,-0.5064315795898438,-0.6547110080718994,-0.1585574895143509,0.40287601947784424,-0.8792572617530823,0.8725411891937256,-0.5235766172409058,0.551956295967102,-1.616973876953125,0.7344838380813599,-2.1006178855895996,0.6636972427368164,0.3945094347000122,-1.370570182800293,0.07480968534946442,-0.19894933700561523,-0.5298669338226318,-0.7848787307739258,-0.6934372186660767,1.0431194305419922,1.1103864908218384,0.5065039396286011,0.2555731236934662,1.686985731124878,-0.7395192384719849,1.5575898885726929,-0.5048502683639526,0.39747563004493713,0.5613957643508911,-1.6452418565750122,-1.4124250411987305,-0.37187397480010986,0.2637156844139099,0.044945698231458664,-0.8186399936676025,-0.3490537405014038,-1.9461230039596558,2.091184377670288,-1.876999855041504,0.02519148588180542,0.08719244599342346,0.16580398380756378,0.3964640200138092,0.5649416446685791,-1.141135573387146,0.4675142168998718,1.0416184663772583,-1.6518527269363403,1.2884585857391357,0.11591645330190659,0.150620698928833,0.7876836061477661,0.09340327233076096,-0.17070399224758148,-1.5061938762664795,-0.24413225054740906,-0.4058179557323456,0.8020711541175842,-0.6087077856063843,-0.03180114924907684,-0.09178069233894348,-1.613738775253296,-0.155301034450531,-0.5883173942565918,0.4585244953632355,-0.6013964414596558,1.2576067447662354,0.4427909851074219,-0.9436922073364258,-1.8521229028701782,0.2701607942581177,-0.5950731635093689,0.5576406717300415,0.9674991369247437,-1.1031535863876343,0.8175444006919861,0.2308807671070099,0.8598411083221436,-0.2858640253543854,1.9551265239715576,0.2581416964530945,-0.654570460319519,0.7529764771461487,-1.4225571155548096,0.16272659599781036,0.09841877222061157]}}
```

## Author

* **Nandan M**


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


