import re
import base64
import json
import time
import random

class dummyrag:
    def __init__(self):
        self.img1 = ""
        self.img2 = ""
        with open('img1.png', 'rb') as file:
            self.img1 = base64.b64encode(file.read()).decode()

        with open('img2.png', 'rb') as file:
            self.img2 = base64.b64encode(file.read()).decode()

        self.history = [
                {"role": "user", "content": "Hi"},
                {"role": "ai", "content": "Hey"},
                {"role": "user", "content": "how are you?"},
                {"role": "ai", "content": "i am **definately not fucking fine**, get lost moron"}
        ]

        self.ipsums = ["""
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec molestie justo massa, efficitur posuere arcu blandit ut. Proin sed gravida lectus, sit amet vestibulum tellus. Maecenas finibus nulla justo, et sodales odio consequat id. Donec semper elit sapien, at tincidunt est sagittis vitae. Sed dignissim tincidunt urna. Donec suscipit lorem sit amet mattis hendrerit. Vivamus sed diam quis purus commodo venenatis vitae accumsan felis. Nunc sed enim enim. Nulla pretium turpis urna, nec rhoncus neque varius nec. Vestibulum in dictum erat.""","""\n\nEtiam eget maximus tellus. In eleifend, ligula non sodales sodales, lectus ipsum condimentum risus, at elementum lorem felis nec mauris. Donec consequat feugiat mi ac tristique. In eu posuere felis, quis ultrices erat. Aliquam tincidunt efficitur dui sit amet dignissim. Sed porta vulputate lorem, non sagittis magna egestas quis. Aenean sit amet lobortis lectus. Pellentesque vel purus hendrerit, mattis felis id, interdum arcu. Etiam vestibulum gravida imperdiet.""","""


                       Etiam fermentum aliquam diam vel egestas. Proin blandit quam eget efficitur dapibus. Aliquam erat volutpat. Curabitur ullamcorper nibh mauris, eget rutrum nunc consectetur eu. Donec interdum nisi non purus convallis pellentesque. Phasellus quis tristique mauris, ac luctus dolor. Praesent eu lobortis velit.""","""

                       Etiam varius ac lorem in scelerisque. Mauris dapibus, enim non viverra placerat, felis mi efficitur nulla, quis imperdiet felis enim nec metus. Integer lobortis lectus a enim feugiat consequat. Donec mi ipsum, facilisis vel turpis id, ullamcorper eleifend augue. Cras pretium nec velit vitae hendrerit. Nunc a fermentum neque. Aliquam ac venenatis odio, ac sollicitudin ante. Etiam nec tempor ex. Donec gravida ipsum orci, sit amet posuere nisi interdum convallis. Integer congue, eros quis posuere porttitor, neque ligula ultrices leo, at lacinia nulla purus ut urna.""","""

                       Morbi sodales et metus id mollis. Pellentesque at blandit leo. Aenean mi mi, mollis sed sem in, tempor egestas mi. Proin in eros vitae neque lacinia faucibus. Vivamus molestie nibh id justo viverra hendrerit. Vestibulum eu sodales velit. Phasellus tincidunt faucibus purus, ac rhoncus ligula maximus id. Morbi massa enim, porta sit amet est vitae, cursus malesuada magna. Fusce sed augue non sapien maximus hendrerit. Donec facilisis magna risus, vitae blandit odio scelerisque eu. """]

        self.ipsum_image = f"""
![](data:image/png;base64,{self.img1})
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

""" + f"""![](data:image/png;base64,{self.img2})""" + """
**Taedis cum** cantus gemitus quid solus intrat
[habentia](#rigidi-nullum-area-respondere) vulnera, facta silet. *Ministro*
Rutulos; oculos pignus, in quam. Huic Est.

- Ante putaret Persephones perire
- Cetera sana
- Habenas terra
- Vestigia gignis

Victima ille aeratae, Gradive mutare in estque crimen requirenti omnem Elateius,
inopes. Currum iacebat inhonestaque haec, hactenus vocibus viaeque leto!

""" + f"""![](data:image/png;base64,{self.img1})""" + """
- Parantem iacentem creditur possint membrana essent
- Proxima tenent
- Adest lumina
- Quod forti
- Nympharum moveat herbida moenia
- Tauri alii durescit

Invocat carinam! Alto et cultusque caduca at viderat irata: tamen per utque
vertice. Sua diesque blanditiae Copia, erat funeris, de aura delubraque quamvis
sive subita. Tuli `wordartBandwidth` nactasque vel, huic paene comas.

""" + f"""![](data:image/png;base64,{self.img2})""" + """

1. Unda Medusae Stygia positamque auras feremus semel
2. Plena attonitos
3. Glomerataque pater properatis et
4. Est non ordine gramina
5. Ei hoc nec remorum ille inrita manabant

Foret numen, eatque *nostris penna nec* spumantia patruo tecta confinia moles.
Illa alumnus, arma illic `ciscRequirements` quoque nympharum ferox est fallaci
auctor, in ibi moriturus, ruit crimine haerebis. Iussus ipse, spatii ipso voce
pinuque proxima; erat."""

        self.files = [
            "Machine Learning.pdf",
            "PYQ.pdf",
            "Syllabus.pdf"
        ]

        self.queries = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Donec molestie justo massa, efficitur posuere arcu blandit ut. Proin sed gravida lectus, sit amet vestibulum tellus. Maecenas finibus nulla justo, et sodales odio consequat id. Donec semper elit sapien, at tincidunt est sagittis vitae. Sed dignissim tincidunt urna. Donec suscipit lorem sit amet mattis hendrerit. Vivamus sed diam quis purus commodo venenatis vitae accumsan felis. Nunc sed enim enim. Nulla pretium turpis urna, nec rhoncus neque varius nec. Vestibulum in dictum erat".split(".")


    def generate(self, prompt, stream: bool = False):
        if not stream:
            yield {
                "type": "final", 
                "status": "DONE", 
                "content": self.ipsums[random.randint(0, len(self.ipsums) - 1)]
            }
        
        else:
            data = {"type": "start", "status": "Generating search query", "content": ""}

            time.sleep(float((random.random() * 10) % 2))
            data = {"type": "query", "status": "Looking for sources", "content": self.queries[random.randint(0, len(self.queries) - 1)]}
            yield data

            time.sleep(float((random.random() * 10) % 2))
            data = {"type": "files", "status": "Formulating response", "content": random.sample(self.files, random.randint(1, 2))}
            yield data

            time.sleep(float((random.random() * 10) % 4))
            data = {"type": "answer", "status": "Inserting Images", "content": self.ipsums[random.randint(0, len(self.ipsums) - 1)]}
            yield data

            time.sleep(float((random.random() * 10) % 4))
            data = {"type": "answer_with_images", "status": "DONE", "content": self.ipsum_image}

            yield data

    def get_history(self) -> dict:
        return self.history
        



