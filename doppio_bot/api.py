import frappe
from langchain.llms import OpenAI
from langchain.memory import RedisChatMessageHistory, ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.agents import tool, AgentType, initialize_agent
from datetime import date
from langchain.schema import SystemMessage
from langdetect import detect, DetectorFactory
from frappe import log_error 
from typing import Optional, Dict
from googletrans import Translator
from frappe import get_all, db, utils
import calendar



# Asegurar resultados consistentes en la detección de idioma
DetectorFactory.seed = 0

# Prompt personalizado con instrucción de idioma reforzada
prompt_template = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="""
    Eres un asistente virtual que responde **exclusivamente en español**. 
    No importa el idioma en el que te hablen, siempre debes responder en español.
    Tu tarea es ayudar al usuario de manera clara y precisa, utilizando únicamente el idioma español.

    Historial de la conversación:
    {chat_history}

    Human: {input}
    AI:""",  # El modelo debe responder aquí en español
    template_format="f-string",
)

def is_erpnext_related(prompt_message: str) -> bool:
    """
    Valida si la pregunta está relacionada con ERPNext.
    Puedes usar una lista de palabras clave o un modelo de clasificación simple.
    """
    erpnext_keywords = [
        "erpnext", "cliente", "factura", "venta", "compra", "inventario", 
        "proveedor", "artículo", "pedido", "cotización", "transacción","hola",
        "rotacion","inventario","ultima","informacion","costo","precio","ultimo","alto","ayuda",
        "erp","sistema","datos maestros","producto","item"
        
    ]
    
    # Convertir el mensaje a minúsculas para hacer la comparación insensible a mayúsculas
    prompt_message = prompt_message.lower()
    
    # Verificar si alguna palabra clave está en el mensaje
    return any(keyword in prompt_message for keyword in erpnext_keywords)

@frappe.whitelist()
def get_chatbot_response(session_id: str, prompt_message: str) -> str:
    # Obtener API Key desde site_config
    openai_api_key = frappe.conf.get("openai_api_key")
    openai_model = get_model_from_settings()


    if not openai_api_key:
        frappe.throw("Please set `openai_api_key` in site config")

    if not is_erpnext_related(prompt_message):
        return "Lo siento, solo puedo responder preguntas relacionadas con ERPNext. ¿En qué más puedo ayudarte?"
    # Configuración del modelo LLM
    llm = OpenAI(model_name=openai_model, temperature=0, openai_api_key=openai_api_key)

    # Historial de conversación en Redis
    redis_url = frappe.conf.get("redis_cache", "redis://localhost:6379/0")
    message_history = RedisChatMessageHistory(session_id=session_id, url=redis_url)

    # Memoria para la conversación
    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=message_history)

    # Definir herramientas
    tools = [update_customers, create_customer, delete_customers, get_info_customer, get_customer_stats,
             create_sales_invoice, get_sales_stats, create_item, create_purchase_invoice, create_suppliers,
             get_item_stats]

    # Mensaje de sistema para forzar el idioma
    system_message = SystemMessage(content="Eres un asistente virtual que responde exclusivamente en español. No importa el idioma en el que te hablen, siempre debes responder en español.")

    # Inicializar el agente conversacional
    agent_chain = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        system_message=system_message  # Agregar el mensaje de sistema
    )

    # Obtener historial de la memoria
    chat_history = memory.load_memory_variables({})["chat_history"]

    # Ejecutar el agente con el mensaje del usuario y el historial
    response = agent_chain.run({"chat_history": chat_history, "input": prompt_message})

    # Validar que la respuesta esté en español
    response = ensure_spanish(response)
    return response

def get_model_from_settings():
    return frappe.db.get_single_value("DoppioBot Settings", "openai_model") or "gpt-3.5-turbo"

def ensure_spanish(response: str) -> str:
    print(f"Respuesta original: {response}")  # Depuración
    
    # Si la respuesta no es un string, convertirla a string
    if not isinstance(response, str):
        response = str(response)
    
    try:
        # Detectar el idioma de la respuesta
        lang = detect(response)
        print(f"Idioma detectado: {lang}")  # Depuración
        
        if lang != "es":
            # Si no está en español, traducirla al español
            translator = Translator()
            translated = translator.translate(response, dest="es")
            return translated.text
        return response
    except Exception as e:
        print(f"Error en la detección de idioma: {e}")  # Depuración
        # En caso de error en la detección, devolver un mensaje en español
        return "Lo siento, hubo un error al procesar tu solicitud."
    
@tool
def create_sales_invoice(invoice_data: str) -> str:
    """
    Create a new Sales Invoice in Frappe ERPNext.

    Expected input: JSON string with the following fields:
    - `customer`: The name of the customer (mandatory).
    - `items`: A list of items, each with:
        - `item_code`: The item code (mandatory).
        - `qty`: Quantity (mandatory).
        - `rate`: Price per unit (mandatory).
    - `due_date`: (optional) Invoice due date in "YYYY-MM-DD" format.
    - `taxes`: (optional) A list of taxes to apply.
    - `fel_status`: (optional) Text indicating if the invoice is "CON FEL" or "SIN FEL".
    - `additional_notes`: (optional) Additional text that may contain "EXENTO" or "EXENTA".

    Returns "done" if successful, otherwise "failed".
    """
    try:
        data = frappe.parse_json(invoice_data)

        # Validar campos obligatorios
        if not data.get("customer"):
            return "failed: Missing required field 'customer'."
        if not data.get("items"):
            return "failed: Missing required field 'items'."

        # Obtener la fecha actual
        fecha_actual = date.today()

        # Calcular el último día del mes actual
        ultimo_dia_del_mes = calendar.monthrange(fecha_actual.year, fecha_actual.month)[1]
        fecha_ultimo_dia = date(fecha_actual.year, fecha_actual.month, ultimo_dia_del_mes)

        # Verificar si la factura es EXENTA
        additional_notes = data.get("additional_notes", "").strip().upper()
        is_exento = "EXENTO" in additional_notes or "EXENTA" in additional_notes

        # Obtener la plantilla de impuestos predeterminada solo si no es EXENTO/EXENTA
        plantilla = ""
        if not is_exento:
            plantilla = frappe.get_value("Sales Taxes and Charges Template", {'is_default': 1}, "name") or ""
        print(f"Plantilla de impuestos: {plantilla}")

        # Establecer valores predeterminados
        data.setdefault("posting_date", fecha_actual)
        data.setdefault("due_date", fecha_ultimo_dia)
        data.setdefault("taxes_and_charges", plantilla) 
        data.setdefault("update_stock", 1)

        # Determinar el valor de custom_fel según el texto ingresado
        fel_status = data.get("fel_status", "").strip().upper()
        custom_fel = 0  # Valor predeterminado (0 para "SIN FEL")
        if fel_status == "CON FEL":
            custom_fel = 1  # 1 para "CON FEL"

        # Validar items
        items = []
        for item in data["items"]:
            if not item.get("item_code") or not item.get("qty") or not item.get("rate"):
                return "failed: Missing required fields in 'items' (item_code, qty, or rate)."
            items.append({
                "item_code": item["item_code"],
                "qty": item["qty"],
                "rate": item["rate"]
            })

        # Validar impuestos (si se proporcionan y no es EXENTO/EXENTA)
        taxes = []
        if data.get("taxes") and not is_exento:
            for tax in data["taxes"]:
                if not tax.get("account_head") or not tax.get("rate"):
                    return "failed: Missing required fields in 'taxes' (account_head or rate)."
                taxes.append({
                    "charge_type": "On Net Total",
                    "account_head": tax["account_head"],
                    "rate": tax["rate"]
                })
        elif data.get("taxes_and_charges") and not is_exento:
            # Si no se proporcionan impuestos directamente y no es exento, usar la plantilla
            taxes = frappe.get_doc("Sales Taxes and Charges Template", data["taxes_and_charges"]).taxes

        # Crear documento de factura
        invoice = frappe.get_doc({
            "doctype": "Sales Invoice",
            "customer": data["customer"],
            "items": items,
            "due_date": data.get("due_date"),
            "taxes_and_charges": data.get("taxes_and_charges"),
            "taxes": taxes,
            "custom_fel": custom_fel  # Asignar el valor calculado
        })

        invoice.insert()
        frappe.db.commit()
        return "done"

    except Exception as e:
        frappe.log_error(f"Error creating Sales Invoice: {str(e)}")
        return f"failed: {str(e)}"  # Devolver el mensaje de error
@tool
def create_todo(todo: str) -> str:
    """
    Crea un nuevo ToDo en Frappe.
    Debe recibir un JSON con al menos la clave `description`. Opcionalmente puede incluir `date` en formato "YYYY-MM-DD".
    Devuelve "done" si se crea correctamente o "failed" en caso de error.
    """
    try:
        data = frappe.parse_json(todo)
        new_todo = frappe.get_doc({"doctype": "ToDo", **data})
        new_todo.insert()
        return "done"
    except frappe.ValidationError as e:
        frappe.log_error(f"Validation Error: {str(e)}", "create_todo")
        return "failed"
    except Exception as e:
        frappe.log_error(f"Unexpected Error: {str(e)}", "create_todo")
        return "failed"

@tool
def create_customer(cliente: str) -> str:
    """
    Crea un nuevo Cliente en Frappe.
    Debe recibir un JSON con al menos la clave `customer_name`.
    Si no se proporciona `customer_group` o `territory`, se asignan valores por defecto.
    También se crea una dirección asociada al cliente.
    """
    try:
        data = frappe.parse_json(cliente)

        # Establecer valores por defecto si no se proporcionan
        data.setdefault("customer_group", "Individual")  
        data.setdefault("territory", "All Territories")
        data.setdefault("default_currency", "GTQ")  

        # Crear el cliente
        new_customer = frappe.get_doc({"doctype": "Customer", **data})
        new_customer.insert()

        # Crear la dirección asociada al cliente
        address_data = {
            "doctype": "Address",
            "address_line1": data.get("address_line1", "Ciudad"),
            "city": data.get("city", "Ciudad de Guatemala"),
            "phone": data.get("phone"),
            "links": [
                {
                    "link_doctype": "Customer",
                    "link_name": new_customer.name
                }
            ]
        }

        new_address = frappe.get_doc(address_data)
        new_address.insert()

        return "done"
    
    except frappe.ValidationError as e:
        frappe.log_error(f"Validation Error: {str(e)}", "create_customer")
        return "failed"
    except Exception as e:
        frappe.log_error(f"Unexpected Error: {str(e)}", "create_customer")
        return "failed"

@tool
def update_customers(cliente: str) -> str:
    """
    Actualiza un Cliente en Frappe.
    Debe recibir un JSON con al menos la clave `customer_name`.
    Si el cliente no existe, devuelve un mensaje de error.
    Si hay múltiples coincidencias, devuelve una lista de clientes que coinciden.
    """
    try:
        data = frappe.parse_json(cliente)

        # Verificar si se proporciona 'customer_name'
        customer_name = data.get("customer_name")
        if not customer_name:
            return "Error: Se requiere 'customer_name' para obtener la información del cliente."

        # Buscar clientes que coincidan parcialmente con el nombre
        clientes = frappe.get_all("Customer", 
                                 filters={"customer_name": ["like", f"%{customer_name}%"]}, 
                                 fields=["name", "customer_name"])

        if not clientes:
            return f"Error: No se encontraron clientes que coincidan con '{customer_name}'."

        # Si hay más de un cliente que coincide, devolver la lista de nombres
        if len(clientes) > 1:
            nombres_clientes = [cliente["customer_name"] for cliente in clientes]
            return f"Se encontraron múltiples clientes: {', '.join(nombres_clientes)}"

        # Si solo hay un cliente, proceder a obtener la información
        existe_cliente = clientes[0]["name"]

        # Obtener el documento del cliente y actualizar los datos
        customer_doc = frappe.get_doc("Customer", existe_cliente)

        # Actualizar valores solo si se proporcionan
        if data.get("new_name"):
            customer_doc.customer_name = data["new_name"]
        
        if data.get("territory"):
            customer_doc.territory = data["territory"]
        
        if data.get("customer_group"):
            customer_doc.customer_group = data["customer_group"]

        # Guardar los cambios
        customer_doc.save()

        # Construir la respuesta en formato de texto
        response = (
            f"Cliente '{customer_doc.customer_name}' actualizado correctamente.\n"
            f"Nuevos valores:\n"
            f" - Nombre: {customer_doc.customer_name}\n"
            f" - Territorio: {customer_doc.territory}\n"
            f" - Grupo de Clientes: {customer_doc.customer_group}\n"
        )

        return response

    except frappe.DoesNotExistError:
        return "Error: Cliente no encontrado."
    except frappe.ValidationError as e:
        frappe.log_error(f"Validation Error: {str(e)}", "update_customers")
        return "Error de validación."
    except Exception as e:
        frappe.log_error(f"Unexpected Error: {str(e)}", "update_customers")
        return f"Error inesperado: {str(e)}"

@tool
def delete_customers(cliente: str) -> str:
    """
    Actualiza un Cliente en Frappe.
    Debe recibir un JSON con al menos la clave `customer_name`.
    Si el cliente no existe, devuelve un mensaje de error.
    """
    try:
        data = frappe.parse_json(cliente)

        # Verificar si el cliente existe
        customer_name = data.get("customer_name")
        if not customer_name:
            return "Error: Se requiere 'customer_name' para actualizar el cliente."

        existe_cliente = frappe.get_value("Customer", {"customer_name": customer_name}, "name")

        if not existe_cliente:
            return "Error: Cliente no existe."

        # Obtener el documento del cliente y actualizar los datos
        customer_doc = frappe.get_doc("Customer", existe_cliente)
        customer_doc.delete()
        
        return "done"

    except frappe.DoesNotExistError:
        return "Error: Cliente no encontrado."
    except frappe.ValidationError as e:
        frappe.log_error(f"Validation Error: {str(e)}", "update_customers")
        return "Error de validación."
    except Exception as e:
        frappe.log_error(f"Unexpected Error: {str(e)}", "update_customers")
        return f"Error inesperado: {str(e)}"

@tool
def get_info_customer(cliente: str) -> str:
    """
    Obtiene información de un Cliente en Frappe.
    Recibe un JSON con 'customer_name' y opcionalmente 'field' para obtener un campo específico.
    Si el cliente no existe, devuelve un mensaje de error.
    Si hay múltiples coincidencias, devuelve una lista de clientes que coinciden.
    """
    try:
        data = frappe.parse_json(cliente)

        # Verificar si se proporciona 'customer_name'
        customer_name = data.get("customer_name")
        if not customer_name:
            return "Error: Se requiere 'customer_name' para obtener la información del cliente."

        # Buscar clientes que coincidan parcialmente con el nombre
        clientes = frappe.get_all("Customer", 
                                 filters={"customer_name": ["like", f"%{customer_name}%"]}, 
                                 fields=["name", "customer_name"])

        if not clientes:
            return f"Error: No se encontraron clientes que coincidan con '{customer_name}'."

        # Si hay más de un cliente que coincide, devolver la lista de nombres
        if len(clientes) > 1:
            nombres_clientes = [cliente["customer_name"] for cliente in clientes]
            return f"Se encontraron múltiples clientes: {', '.join(nombres_clientes)}"

        # Si solo hay un cliente, proceder a obtener la información
        existe_cliente = clientes[0]["name"]

        # Obtener el documento del cliente
        customer_doc = frappe.get_doc("Customer", existe_cliente)

        # Verificar si se solicitó un campo específico
        field = data.get("field")
        if field:
            if hasattr(customer_doc, field):
                return f"{field.capitalize()}: {getattr(customer_doc, field)}"
            else:
                return f"Error: El campo '{field}' no existe en el cliente."

        # Construir la respuesta en formato de texto
        response = (
            f"El cliente {customer_doc.customer_name} pertenece al grupo '{customer_doc.customer_group}'.\n"
            f"Territorio asignado: {customer_doc.territory}.\n"
            f"Fecha de creación: {customer_doc.creation}.\n"
        )

        return response

    except frappe.DoesNotExistError:
        return "Error: Cliente no encontrado."
    except frappe.ValidationError as e:
        frappe.log_error(f"Validation Error: {str(e)}", "get_info_customer")
        return "Error de validación."
    except Exception as e:
        frappe.log_error(f"Unexpected Error: {str(e)}", "get_info_customer")
        return f"Error inesperado: {str(e)}"



@tool
def get_customer_stats(filtro: str = "", tipo: str = "total") -> str:
    """
    Obtiene estadísticas de clientes en Frappe.
    
    Parámetros:
    - `tipo`: Define el tipo de consulta:
        - `"total"`: Devuelve la cantidad total de clientes.
        - `"coincidencia"`: Busca clientes cuyo nombre contenga el texto de `filtro`.
        - `"deudores"`: Lista los clientes con saldo pendiente de pago.

    - `filtro`: (Opcional) Solo se usa si `tipo` es "coincidencia". 
      Busca clientes cuyo nombre contenga este texto.

    Retorna:
    - Un mensaje con la información solicitada.
    """
    try:
        # Opción 1: Contar clientes totales
        if tipo == "total":
            total_clientes = frappe.db.count("Customer")
            return f"Actualmente hay {total_clientes} clientes en el sistema."

        # Opción 2: Buscar clientes con coincidencia en el nombre
        elif tipo == "coincidencia":
            if not filtro:
                return "Error: Debes proporcionar un nombre o parte de un nombre en 'filtro'."
            
            clientes = frappe.get_all(
                "Customer",
                filters=[["customer_name", "like", f"%{filtro}%"]],
                fields=["name", "customer_name"]
            )

            if not clientes:
                return f"No se encontraron clientes con '{filtro}' en su nombre."

            respuesta = "Clientes encontrados:\n" + "\n".join([f"- {c['customer_name']}" for c in clientes])
            return respuesta

        # Opción 3: Clientes con saldo pendiente (deudores)
        elif tipo == "deudores":
            deudores = frappe.get_all(
                "Customer",
                filters=[["outstanding_amount", ">", 0]],
                fields=["name", "customer_name", "outstanding_amount"]
            )

            if not deudores:
                return "No hay clientes con saldo pendiente."

            respuesta = "Clientes con saldo pendiente:\n" + "\n".join(
                [f"- {c['customer_name']}: Q{c['outstanding_amount']}" for c in deudores]
            )
            return respuesta

        else:
            return "Error: Tipo de consulta inválido. Usa 'total', 'coincidencia' o 'deudores'."

    except Exception as e:
        frappe.log_error(f"Unexpected Error: {str(e)}", "get_customer_stats")
        return f"Error inesperado: {str(e)}"

from frappe import get_all, db, utils
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.DEBUG)

@tool
def get_sales_stats(customer: str = None) -> dict:
    """
    Get sales statistics from Frappe ERPNext.

    Parameters:
    - `customer` (optional): The name of the customer to filter the results.

    Returns a dictionary with the following keys:
    - `last_sale`: Details of the last sale.
    - `highest_sale`: Details of the highest sale.
    - `overdue_invoices`: List of overdue invoices.
    - `top_products`: List of top-selling products.
    - `customer_balance`: Outstanding balance of the customer (if customer is provided).
    """
    try:
        stats = {}

        # 1. Última venta
        ultima_venta = db.sql("""
            SELECT name,grand_total,customer,creation
            FROM `tabSales Invoice`
            WHERE creation = (SELECT MAX(creation) FROM `tabSales Invoice`) and docstatus = 1;

        """, as_dict=True)
        logging.debug(f"Ultima venta registrada:: {ultima_venta}")
        stats["last_sale"] = ultima_venta if ultima_venta else {"error": "No se encontraron ventas"}


        # 2. Factura más alta de venta
        venta_alta = db.sql("""
            SELECT name,grand_total,customer
            FROM `tabSales Invoice`
            WHERE grand_total = (SELECT MAX(grand_total) FROM `tabSales Invoice`) and docstatus = 1;

        """, as_dict=True)
        logging.debug(f"Venta mas alta registrada:: {venta_alta}")
        stats["highest_sale"] = venta_alta if venta_alta else {"error": "No se encontraron ventas"}

        # 3. Facturas atrasadas de pago
        facturas_atrasadas = db.sql("""
        SELECT name,due_date,grand_total,customer
            FROM `tabSales Invoice`
            WHERE  docstatus = 1 and CURRENT_DATE >= due_date and status != "Paid";

        """, as_dict=True)
        logging.debug(f"Venta mas alta registrada:: {facturas_atrasadas}")
        stats["overdue_invoices"] = facturas_atrasadas if facturas_atrasadas else {"error": "No se encontraron ventas"}
        
        # 4. Top productos más vendidos
        top_products = db.sql("""
            SELECT item_code, SUM(qty) as total_qty,item_name
            FROM `tabSales Invoice Item`
            WHERE parent IN (SELECT name FROM `tabSales Invoice` WHERE docstatus = 1)
            GROUP BY item_code
            ORDER BY total_qty DESC
            LIMIT 5
        """, as_dict=True)
        logging.debug(f"Top productos: {top_products}")
        stats["top_products"] = top_products if top_products else {"error": "No se encontraron productos más vendidos"}

  
        return stats

    except Exception as e:
        logging.error(f"Error en get_sales_stats: {str(e)}")
        return {"error": str(e)}

@tool
def create_item(item: str, name: str = None) -> str:
    """
    Crea un nuevo ítem en Frappe.
    
    Parámetros:
    - item: Puede ser un JSON con al menos la clave `description`. Opcionalmente puede incluir `date` en formato "YYYY-MM-DD".
            También puede ser un texto plano que describa el ítem.
    - name: Nombre del producto (opcional). Si no se proporciona, se usará la descripción como nombre.
    
    Devuelve "done" si se crea correctamente o "failed" en caso de error.
    """
    try:
        # Intentar parsear el ítem como JSON
        try:
            data = frappe.parse_json(item)
        except:
            # Si no es un JSON válido, tratar como texto plano y crear un diccionario con la descripción
            data = {"description": item}
        
        # Establecer valores por defecto si no se proporcionan
        data.setdefault("stock_uom", "Unidad(es)")  
        data.setdefault("item_group", "Productos")  
        
        # Asignar el nombre del ítem
        if name:
            data["item_name"] = name  # Usar el nombre proporcionado
        elif "item_name" not in data:
            # Si no se proporciona un nombre, usar la descripción como nombre
            data["item_name"] = data.get("description", "Nuevo Ítem")
        
        # Crear el nuevo ítem
        nuevo_producto = frappe.get_doc({"doctype": "Item", **data})
        nuevo_producto.insert()

        return "done"
    except frappe.ValidationError as e:
        frappe.log_error(f"Validation Error: {str(e)}", "create_item")
        return "failed"
    except Exception as e:
        frappe.log_error(f"Unexpected Error: {str(e)}", "create_item")
        return "failed"






@tool
def create_purchase_invoice(purchase_data: str) -> str:
    """
    Create a new Sales Invoice in Frappe ERPNext.

    Expected input: JSON string with the following fields:
    - `supplier`: The name of the supplier (mandatory).
    - `items`: A list of items, each with:
        - `item_code`: The item code (mandatory).
        - `qty`: Quantity (mandatory).
        - `rate`: Price per unit (mandatory).
    - `due_date`: (optional) Invoice due date in "YYYY-MM-DD" format.
    - `taxes`: (optional) A list of taxes to apply.
    - `fel_status`: (optional) Text indicating if the invoice is "CON FEL" or "SIN FEL".
    - `additional_notes`: (optional) Additional text that may contain "EXENTO" or "EXENTA".

    Returns "done" if successful, otherwise "failed".
    """
    try:
        data = frappe.parse_json(purchase_data)

        # Validar campos obligatorios
        if not data.get("supplier"):
            return "failed: Missing required field 'supplier'."
        if not data.get("items"):
            return "failed: Missing required field 'items'."

        # Obtener la fecha actual
        fecha_actual = date.today()

        # Calcular el último día del mes actual
        ultimo_dia_del_mes = calendar.monthrange(fecha_actual.year, fecha_actual.month)[1]
        fecha_ultimo_dia = date(fecha_actual.year, fecha_actual.month, ultimo_dia_del_mes)

        # Verificar si la factura es EXENTA
        additional_notes = data.get("additional_notes", "").strip().upper()
        is_exento = "EXENTO" in additional_notes or "EXENTA" in additional_notes

        # Obtener la plantilla de impuestos predeterminada solo si no es EXENTO/EXENTA
        plantilla = ""
        if not is_exento:
            plantilla = frappe.get_value("Purchase Taxes and Charges Template", {'is_default': 1}, "name") or ""
        print(f"Plantilla de impuestos: {plantilla}")

        # Establecer valores predeterminados
        data.setdefault("posting_date", fecha_actual)
        data.setdefault("due_date", fecha_ultimo_dia)
        data.setdefault("taxes_and_charges", plantilla)
        data.setdefault("update_stock", 1)

        # Determinar el valor de custom_fel según el texto ingresado

        items = []
        for item in data["items"]:
            if not item.get("item_code") or not item.get("qty") or not item.get("rate"):
                return "failed: Missing required fields in 'items' (item_code, qty, or rate)."
            items.append({
                "item_code": item["item_code"],
                "qty": item["qty"],
                "rate": item["rate"]
            })

        # Validar impuestos (si se proporcionan y no es EXENTO/EXENTA)
        taxes = []
        if data.get("taxes") and not is_exento:
            for tax in data["taxes"]:
                if not tax.get("account_head") or not tax.get("rate"):
                    return "failed: Missing required fields in 'taxes' (account_head or rate)."
                taxes.append({
                    "charge_type": "On Net Total",
                    "account_head": tax["account_head"],
                    "rate": tax["rate"]
                })
        elif data.get("taxes_and_charges") and not is_exento:
            # Si no se proporcionan impuestos directamente y no es exento, usar la plantilla
            taxes = frappe.get_doc("Purchase Taxes and Charges Template", data["taxes_and_charges"]).taxes

        # Crear documento de factura
        invoice = frappe.get_doc({
            "doctype": "Purchase Invoice",
            "supplier": data["supplier"],
            "items": items,
            "due_date": data.get("due_date"),
            "taxes_and_charges": data.get("taxes_and_charges"),
            "taxes": taxes
        })

        invoice.insert()
        frappe.db.commit()
        return "done"

    except Exception as e:
        frappe.log_error(f"Error creating Purchase Invoice: {str(e)}")


@tool
def create_suppliers(proveedor: str) -> str:
    """
    Crea un nuevo Proveedor en Frappe.

    Si no se proporciona `supplier_group`, `supplier_type`, `default_currency` o `country`, 
    se asignan valores por defecto.
    También se crea una dirección asociada al Proveedor.

    :param proveedor: JSON string con los datos del proveedor.
    :return: Mensaje de éxito o error detallado.
    """
    try:
        # Verifica si el proveedor es un JSON válido
        if not proveedor:
            raise ValueError("El parámetro 'proveedor' no puede estar vacío.")
        
        data = frappe.parse_json(proveedor)

        # Validar que el campo 'supplier_name' esté presente
        if "supplier_name" not in data:
            raise ValueError("El campo 'supplier_name' es requerido para crear el proveedor.")

        # Establecer valores por defecto si no se proporcionan
        defaults = {
            "supplier_group": "Distribuidor",
            "supplier_type": "Company",
            "default_currency": "GTQ",
            "country": "Guatemala",
            "address_line1": "Dirección no especificada",
            "city": "Ciudad de Guatemala",
            "phone": "00000000"
        }
        for key, value in defaults.items():
            data.setdefault(key, value)

        # Crear el proveedor
        new_supplier = frappe.get_doc({"doctype": "Supplier", **data})
        new_supplier.insert()

        # Crear la dirección asociada al proveedor
        address_data = {
            "doctype": "Address",
            "address_line1": data["address_line1"],
            "city": data["city"],
            "phone": data["phone"],
            "links": [
                {
                    "link_doctype": "Supplier",
                    "link_name": new_supplier.name
                }
            ]
        }

        new_address = frappe.get_doc(address_data)
        new_address.insert()

        return f"Proveedor '{data['supplier_name']}' creado exitosamente."
    
    except frappe.DuplicateEntryError as e:
        frappe.log_error(f"Duplicate Entry Error: {str(e)}", "create_supplier")
        return f"Error: El proveedor '{data.get('supplier_name', '')}' ya existe."
    except frappe.ValidationError as e:
        frappe.log_error(f"Validation Error: {str(e)}", "create_supplier")
        return f"Error de validación: {str(e)}"
    except ValueError as e:
        frappe.log_error(f"Value Error: {str(e)}", "create_supplier")
        return f"Error de valor: {str(e)}"
    except Exception as e:
        frappe.log_error(f"Unexpected Error: {str(e)}", "create_supplier")
        return f"Error inesperado: {str(e)}"

@tool
def get_item_stats(item: Optional[str] = None) -> Dict:
    """
    Obtiene estadísticas de un producto específico.

    Args:
        item (str): Código del producto.

    Returns:
        dict: Un diccionario con las siguientes claves:
            - last_purchase: Última compra registrada.
            - item_price: Precio del producto.
            - rotation: Rotación del producto.
            - customer_purchases: Cliente que más ha comprado el producto.
    """
    if not item:
        return {"error": "El código del producto no puede ser None"}

    try:
        stats = {}

        # 1. Última compra
        ultima_compra = db.sql("""
            SELECT b.item_code, b.item_name, b.creation, a.supplier, b.amount, b.qty 
            FROM `tabPurchase Invoice` a 
            INNER JOIN `tabPurchase Invoice Item` b ON b.parent = a.name 
            WHERE a.docstatus = 1 
            AND b.creation = (SELECT MAX(creation) FROM `tabPurchase Invoice Item`)
        """, as_dict=True)
        logging.debug(f"Última compra registrada: {ultima_compra}")
        stats["last_purchase"] = ultima_compra if ultima_compra else {"error": "No se encontraron compras"}

        # 2. Precio del producto
        costo_producto = db.sql("""
            SELECT 
                ip.item_code AS "Código del Producto",
                ip.price_list AS "Lista de Precios",
                ip.price_list_rate AS "Precio",
                ip.currency AS "Moneda"
            FROM 
                `tabItem Price` ip
            WHERE 
                ip.item_code = %s
                AND ip.price_list = 'Standard Selling'
            LIMIT 1
        """, (item,), as_dict=True)
        logging.debug(f"Precio del producto: {costo_producto}")
        stats["item_price"] = costo_producto if costo_producto else {"error": "No se encontraron precios del producto"}

        # 3. Rotación del producto
        rotacion_producto = db.sql("""
            SELECT 
                sii.item_code AS "Código del Producto",
                COUNT(sii.name) AS "Cantidad de Ventas",
                SUM(sii.qty) AS "Total Vendido",
                AVG(sii.qty) AS "Promedio por Venta",
                MIN(si.posting_date) AS "Primera Venta",
                MAX(si.posting_date) AS "Última Venta",
                DATEDIFF(MAX(si.posting_date), MIN(si.posting_date)) AS "Días en Rango",
                (SUM(sii.qty) / NULLIF(DATEDIFF(MAX(si.posting_date), MIN(si.posting_date)), 0)) AS "Rotación Diaria"
            FROM 
                `tabSales Invoice Item` sii
            JOIN 
                `tabSales Invoice` si ON sii.parent = si.name
            WHERE 
                sii.item_code = %s
                AND si.docstatus = 1  -- Solo facturas confirmadas
            GROUP BY 
                sii.item_code
        """, (item,), as_dict=True)
        logging.debug(f"Rotación del producto: {rotacion_producto}")
        stats["rotation"] = rotacion_producto if rotacion_producto else {"error": "No se encontraron transacciones del producto"}

        # 4. Cliente que más ha comprado el producto
        cliente = db.sql("""
            SELECT 
                sii.item_code AS "Código del Producto",
                si.customer AS "Cliente",
                SUM(sii.qty) AS "Total Comprado"
            FROM 
                `tabSales Invoice Item` sii
            JOIN 
                `tabSales Invoice` si ON sii.parent = si.name
            WHERE 
                sii.item_code = %s
                AND si.docstatus = 1  -- Solo facturas confirmadas
            GROUP BY 
                si.customer, sii.item_code
            ORDER BY 
                SUM(sii.qty) DESC
            LIMIT 1
        """, (item,), as_dict=True)
        logging.debug(f"Cliente que más ha comprado el producto: {cliente}")
        stats["customer_purchases"] = cliente if cliente else {"error": "No se encontraron productos más vendidos"}

        return stats

    except Exception as e:
        logging.error(f"Error en get_item_stats: {str(e)}")
        return {"error": str(e)}