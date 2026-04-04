import random
import time
import json
import base64

from flask import Flask, jsonify, request, stream_with_context, Response
from flask_cors import CORS

img1 = ""
img2 = ""
with open('img1.png', 'rb') as file:
    img1 = base64.b64encode(file.read()).decode()

with open('img2.png', 'rb') as file:
    img2 = base64.b64encode(file.read()).decode()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

ipsums = ["""
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec molestie justo massa, efficitur posuere arcu blandit ut. Proin sed gravida lectus, sit amet vestibulum tellus. Maecenas finibus nulla justo, et sodales odio consequat id. Donec semper elit sapien, at tincidunt est sagittis vitae. Sed dignissim tincidunt urna. Donec suscipit lorem sit amet mattis hendrerit. Vivamus sed diam quis purus commodo venenatis vitae accumsan felis. Nunc sed enim enim. Nulla pretium turpis urna, nec rhoncus neque varius nec. Vestibulum in dictum erat.""","""

Etiam eget maximus tellus. In eleifend, ligula non sodales sodales, lectus ipsum condimentum risus, at elementum lorem felis nec mauris. Donec consequat feugiat mi ac tristique. In eu posuere felis, quis ultrices erat. Aliquam tincidunt efficitur dui sit amet dignissim. Sed porta vulputate lorem, non sagittis magna egestas quis. Aenean sit amet lobortis lectus. Pellentesque vel purus hendrerit, mattis felis id, interdum arcu. Etiam vestibulum gravida imperdiet.""","""

Etiam fermentum aliquam diam vel egestas. Proin blandit quam eget efficitur dapibus. Aliquam erat volutpat. Curabitur ullamcorper nibh mauris, eget rutrum nunc consectetur eu. Donec interdum nisi non purus convallis pellentesque. Phasellus quis tristique mauris, ac luctus dolor. Praesent eu lobortis velit.""","""

Etiam varius ac lorem in scelerisque. Mauris dapibus, enim non viverra placerat, felis mi efficitur nulla, quis imperdiet felis enim nec metus. Integer lobortis lectus a enim feugiat consequat. Donec mi ipsum, facilisis vel turpis id, ullamcorper eleifend augue. Cras pretium nec velit vitae hendrerit. Nunc a fermentum neque. Aliquam ac venenatis odio, ac sollicitudin ante. Etiam nec tempor ex. Donec gravida ipsum orci, sit amet posuere nisi interdum convallis. Integer congue, eros quis posuere porttitor, neque ligula ultrices leo, at lacinia nulla purus ut urna.""","""

Morbi sodales et metus id mollis. Pellentesque at blandit leo. Aenean mi mi, mollis sed sem in, tempor egestas mi. Proin in eros vitae neque lacinia faucibus. Vivamus molestie nibh id justo viverra hendrerit. Vestibulum eu sodales velit. Phasellus tincidunt faucibus purus, ac rhoncus ligula maximus id. Morbi massa enim, porta sit amet est vitae, cursus malesuada magna. Fusce sed augue non sapien maximus hendrerit. Donec facilisis magna risus, vitae blandit odio scelerisque eu. """
]

ipsum_image = f"""
![](data:image/png;base64,{img1})
""" + """# Rigidi nullum area respondere

*Lorem markdownum ponit* bacchantum inter piscosamque soceri penna. Aetna
membraque dixit, inmortalis terram, mora, loqui Ausonium. Credit Maeonias dedit
ad sumite *stimulosque* veste mulcendaque nigra frater tu.

```python
    if (googlePersonalText) {
        mail.pipeline_memory_ring = -3 + printer;
        friendly_default(volumePingModifier, in_media_smishing, 4);
    } else {
        ip_path(bookmark_import_raw, 3);
        xsltDlc.directoryPopFriend(scrollingMacro, xmp_bar + dSnmp,
                system_digital);
    }
    overclockingComputing.exabyte(4);
    cardFriend.vci.mac(gateway_leaf, rpmIsaLearning(architectureHard),
            workstation_offline);
    vdu_spooling = rpcCableMotherboard + hypermedia_raid_default;
    input_dual_constant /= hardware;
```

""" + f"""![](data:image/png;base64,{img2})""" + """
**Taedis cum** cantus gemitus quid solus intrat
[habentia](#rigidi-nullum-area-respondere) vulnera, facta silet. *Ministro*
Rutulos; oculos pignus, in quam. Huic Est.

- Ante putaret Persephones perire
- Cetera sana
- Habenas terra
- Vestigia gignis

Victima ille aeratae, Gradive mutare in estque crimen requirenti omnem Elateius,
inopes. Currum iacebat inhonestaque haec, hactenus vocibus viaeque leto!

""" + f"""![](data:image/png;base64,{img1})""" + """
- Parantem iacentem creditur possint membrana essent
- Proxima tenent
- Adest lumina
- Quod forti
- Nympharum moveat herbida moenia
- Tauri alii durescit

Invocat carinam! Alto et cultusque caduca at viderat irata: tamen per utque
vertice. Sua diesque blanditiae Copia, erat funeris, de aura delubraque quamvis
sive subita. Tuli `wordartBandwidth` nactasque vel, huic paene comas.

""" + f"""![](data:image/png;base64,{img2})""" + """

1. Unda Medusae Stygia positamque auras feremus semel
2. Plena attonitos
3. Glomerataque pater properatis et
4. Est non ordine gramina
5. Ei hoc nec remorum ille inrita manabant

Foret numen, eatque *nostris penna nec* spumantia patruo tecta confinia moles.
Illa alumnus, arma illic `ciscRequirements` quoque nympharum ferox est fallaci
auctor, in ibi moriturus, ruit crimine haerebis. Iussus ipse, spatii ipso voce
pinuque proxima; erat."""

files = [
    "Machine Learning.pdf",
    "PYQ.pdf",
    "Syllabus.pdf"
]

queries = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec molestie justo massa, efficitur posuere arcu blandit ut. Proin sed gravida lectus, sit amet vestibulum tellus. Maecenas finibus nulla justo, et sodales odio consequat id. Donec semper elit sapien, at tincidunt est sagittis vitae. Sed dignissim tincidunt urna. Donec suscipit lorem sit amet mattis hendrerit. Vivamus sed diam quis purus commodo venenatis vitae accumsan felis. Nunc sed enim enim. Nulla pretium turpis urna, nec rhoncus neque varius nec. Vestibulum in dictum erat".split(".")



@app.route('/')
def root():
    return "Api is working"


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"type": "error", "content": "Unsupported Media Type"}), 415

    if not data.get('prompt', None):
        return jsonify({"type": "error", "content": "invalid request body. prompt not found"})

    time.sleep(random.randint(2, 5))

    return jsonify({
        "type": "ai", 
        "status": "DONE", 
        "content": ipsums[random.randint(0, len(ipsums) - 1)]
    })


@app.route('/stream_g', methods=['GET'])
def stream():
    prompt = request.args.get("prompt", None)
    if not prompt:
        err = json.dumps({"type": "error", "content": "Please specify a prompt"})
        def resp(): 
            yield f"data: {err}\n\n"

        return Response(
            stream_with_context(resp()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )

    def generator(prompt):
        data = {"type": "start", "status": "Generating search query", "content": ""}

        time.sleep(float((random.random() * 10) % 2))
        data = {"type": "query", "status": "Looking for sources", "content": queries[random.randint(0, len(queries) - 1)]}
        yield f"data: {json.dumps(data)}\n\n"

        time.sleep(float((random.random() * 10) % 2))
        data = {"type": "files", "status": "Formulating response", "content": random.sample(files, random.randint(1, 2))}
        yield f"data: {json.dumps(data)}\n\n"

        time.sleep(float((random.random() * 10) % 4))
        data = {"type": "answer", "status": "Inserting Images", "content": ipsums[random.randint(0, len(ipsums) - 1)]}
        yield f"data: {json.dumps(data)}\n\n"

        time.sleep(float((random.random() * 10) % 4))
        data = {"type": "answer_with_images", "status": "DONE", "content": ipsum_image}
        yield f"data: {json.dumps(data)}\n\n"

    return Response(
        stream_with_context(generator(prompt)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
         
@app.route('/stream', methods=['POST'])
def stream_post():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"type": "error", "content": "Unsupported Media Type"}), 415

    prompt = data.get('prompt')
    file = data.get('file')

    if not prompt:
        return jsonify({"type": "error", "content": "No prompt found"})

    def generator():
        data = {"type": "start", "status": "Generating search query", "content": ""}

        time.sleep(float((random.random() * 10) % 2))
        data = {"type": "query", "status": "Looking for sources", "content": queries[random.randint(0, len(queries) - 1)]}
        yield f"data: {json.dumps(data)}\n\n"

        time.sleep(float((random.random() * 10) % 2))
        data = {"type": "files", "status": "Formulating response", "content": random.sample(files, random.randint(1, 2))}
        yield f"data: {json.dumps(data)}\n\n"

        time.sleep(float((random.random() * 10) % 4))
        data = {"type": "answer", "status": "Inserting Images", "content": ipsums[random.randint(0, len(ipsums) - 1)]}
        yield f"data: {json.dumps(data)}\n\n"

        time.sleep(float((random.random() * 10) % 4))
        data = {"type": "answer_with_images", "status": "DONE", "content": ipsum_image}
        yield f"data: {json.dumps(data)}\n\n"

    return Response(
        stream_with_context(generator()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

# Disable Caching
@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__ == "__main__":
    app.run(host="::", port=31337, debug=False)
