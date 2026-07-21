Este diretório contém o código fonte e os artefatos de implementação do projeto.
- AndroidApp_V_0.1/ : Código fonte completo da aplicação nativa Android (Kotlin + Jetpack Compose + MediaPipe + Firebase).
- POC_Python/       : Protótipo de visão computacional em Python com estimativa de pose e regras cinemáticas experimentais.

Instrucoes de Execucao:
Devido a arquitetura baseada em edge computing, a aplicacao foi desenhada para correr num smartphone fisico em vez de um emulador virtual do Android Studio. Os modelos de Computer Vision dependem de arquiteturas hardware-especificas (ARM) que geram falhas em emuladores x86.

Para testar a aplicacao:
1. Ative as "Opcoes de Programador" e a "Depuracao USB" no seu smartphone Android.
2. Ligue o smartphone fisicamente por cabo USB a maquina onde tem o projeto aberto.
3. No topo do Android Studio, selecione o seu smartphone fisico na lista de dispositivos.
4. Execute (Run) o projeto para instalar e correr a aplicacao nativamente.
